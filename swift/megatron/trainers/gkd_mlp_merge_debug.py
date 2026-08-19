# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import inspect
import os

import torch

from swift.utils import get_logger


logger = get_logger()


class GKDMLPMergeIsolation:
    """Capture the Layer-N MLP residual merge and replay the complete MLP VJP."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.mode = os.getenv('SWIFT_GKD_MLP_MERGE_MODE', '').lower()
        self.output_dir = os.getenv('SWIFT_GKD_MLP_MERGE_DIR')
        self.layer_target = os.getenv(
            'SWIFT_GKD_MLP_MERGE_LAYER_TARGET', 'decoder.layers.27')
        self.mlp_target = f'{self.layer_target}.mlp'
        self.fc1_target = f'{self.mlp_target}.linear_fc1'
        self.step = int(os.getenv('SWIFT_GKD_MLP_MERGE_STEP', '0'))
        self.micro_batch = int(os.getenv('SWIFT_GKD_MLP_MERGE_MICRO_BATCH', '0'))
        self.tag = os.getenv('SWIFT_GKD_MLP_MERGE_TAG', '').lower()
        self.x_file = os.getenv('SWIFT_GKD_MLP_MERGE_X_FILE')
        self.dout_file = os.getenv('SWIFT_GKD_MLP_MERGE_DOUT_FILE')
        self.parameter_file = os.getenv('SWIFT_GKD_MLP_MERGE_PARAMETER_FILE')
        self.handles = []
        self.mlp_module = None
        self.payload = None
        self._sources = {}
        self._mlp_call_count = 0
        self._shared_input_hook_registered = False
        self._seen_output_gradients = set()

        if self.mode not in {'', 'capture', 'replay'}:
            raise ValueError(
                'SWIFT_GKD_MLP_MERGE_MODE must be empty, "capture", or "replay".')
        if not self.mode:
            return
        if not self.output_dir:
            raise ValueError('SWIFT_GKD_MLP_MERGE_DIR is required for MLP merge isolation.')
        if self.step < 0 or self.micro_batch < 0:
            raise ValueError('MLP merge isolation step and micro-batch must be non-negative.')
        if self.mode == 'replay':
            missing = [
                name for name, value in (
                    ('SWIFT_GKD_MLP_MERGE_X_FILE', self.x_file),
                    ('SWIFT_GKD_MLP_MERGE_DOUT_FILE', self.dout_file),
                ) if not value
            ]
            if missing:
                raise ValueError(f'MLP merge replay requires: {missing}.')
            self.parameter_file = self.parameter_file or self.x_file

    @property
    def enabled(self):
        return bool(self.mode)

    @property
    def output_path(self):
        tag = self.tag or self.mode
        return os.path.join(self.output_dir, f'mlp_merge_{tag}.pt')

    @staticmethod
    def _first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, (tuple, list)):
            values = value
        else:
            return None
        for item in values:
            tensor = GKDMLPMergeIsolation._first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    @staticmethod
    def _named_tensor_leaves(value, prefix='output'):
        if torch.is_tensor(value):
            return [(prefix, value)]
        if isinstance(value, dict):
            result = []
            for key, item in value.items():
                result.extend(GKDMLPMergeIsolation._named_tensor_leaves(
                    item, f'{prefix}.{key}'))
            return result
        if isinstance(value, (tuple, list)):
            result = []
            for index, item in enumerate(value):
                result.extend(GKDMLPMergeIsolation._named_tensor_leaves(
                    item, f'{prefix}.{index}'))
            return result
        return []

    @staticmethod
    def _identity(tensor):
        value = tensor.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }

    def _context(self):
        for context in (
                self.trainer._operator_debug_context,
                self.trainer._backward_debug_context):
            if (context is not None
                    and context['step'] == self.step
                    and context['micro_batch'] == self.micro_batch):
                return context
        return None

    def _load_source(self, path):
        if path not in self._sources:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'MLP merge replay source not found: {path}')
            source = torch.load(path, map_location='cpu', weights_only=True)
            if source.get('mlp_target') != self.mlp_target:
                raise ValueError(
                    f'MLP replay source target {source.get("mlp_target")} does not match '
                    f'{self.mlp_target}: {path}')
            self._sources[path] = source
        return self._sources[path]

    def _replace_first_tensor(self, args, kwargs, replacement):
        args = list(args)
        for index, value in enumerate(args):
            if torch.is_tensor(value):
                args[index] = replacement
                return tuple(args), kwargs
        kwargs = dict(kwargs)
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = replacement
                return tuple(args), kwargs
        raise RuntimeError(f'MLP target received no replaceable tensor: {self.mlp_target}')

    def _new_payload(self, module, input_tensor, runtime_parameters):
        return {
            'mode': self.mode,
            'tag': self.tag,
            'step': self.step,
            'micro_batch': self.micro_batch,
            'layer_target': self.layer_target,
            'mlp_target': self.mlp_target,
            'fc1_target': self.fc1_target,
            'module_type': type(module).__name__,
            'module_signature': str(inspect.signature(module.forward)),
            'input': input_tensor.detach().cpu().contiguous(),
            'input_identity': self._identity(input_tensor),
            'parameters': {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in runtime_parameters.items()
            },
            'parameter_identities': {
                name: self._identity(parameter)
                for name, parameter in runtime_parameters.items()
            },
            'forward_outputs': {},
            'output_gradients': {},
            'layer_output_gradient': None,
            'shared_input_gradient': None,
            'mlp_local_input_gradient': None,
            'mlp_local_input_gradient_source': None,
            'mlp_forward_call_count': 0,
        }

    def _mlp_pre_hook(self, name):

        def hook(module, args, kwargs):
            if self._context() is None:
                return args, kwargs
            runtime_input = self._first_tensor(args)
            if runtime_input is None:
                runtime_input = self._first_tensor(kwargs)
            if runtime_input is None:
                raise RuntimeError(f'MLP merge target received no tensor input: {name}')
            runtime_parameters = dict(module.named_parameters())

            if self.mode == 'capture':
                isolated_input = runtime_input
            else:
                x_source = self._load_source(self.x_file)
                parameter_source = self._load_source(self.parameter_file)
                common_input = x_source['input']
                if tuple(common_input.shape) != tuple(runtime_input.shape):
                    raise ValueError(
                        f'MLP replay input shape {tuple(common_input.shape)} does not match '
                        f'runtime shape {tuple(runtime_input.shape)}.')
                common_parameters = parameter_source.get('parameters', {})
                if set(common_parameters) != set(runtime_parameters):
                    raise ValueError(
                        f'MLP replay parameter names differ: source={sorted(common_parameters)}, '
                        f'runtime={sorted(runtime_parameters)}.')
                with torch.no_grad():
                    for parameter_name, parameter in runtime_parameters.items():
                        common_parameter = common_parameters[parameter_name]
                        if tuple(common_parameter.shape) != tuple(parameter.shape):
                            raise ValueError(
                                f'MLP parameter {parameter_name} shape mismatch: '
                                f'source={tuple(common_parameter.shape)}, runtime={tuple(parameter.shape)}.')
                        parameter.copy_(common_parameter.to(
                            device=parameter.device, dtype=parameter.dtype))
                isolated_input = common_input.to(
                    device=runtime_input.device,
                    dtype=runtime_input.dtype,
                ).detach().requires_grad_(True)
                args, kwargs = self._replace_first_tensor(args, kwargs, isolated_input)

            if self.payload is None:
                self.payload = self._new_payload(module, isolated_input, runtime_parameters)
                if self.mode == 'replay':
                    self.payload.update({
                        'x_source_file': os.path.abspath(self.x_file),
                        'dout_source_file': os.path.abspath(self.dout_file),
                        'parameter_source_file': os.path.abspath(self.parameter_file),
                    })
            self._mlp_call_count += 1
            self.payload['mlp_forward_call_count'] = self._mlp_call_count

            # The first native MLP input is shared with the residual path. Its tensor hook
            # receives the sum of residual and local MLP gradients. A checkpoint recompute
            # may call this pre-hook again with an internal detached tensor, so only hook call 1.
            if (self.mode == 'capture' and not self._shared_input_hook_registered
                    and isolated_input.requires_grad):

                def shared_input_hook(gradient):
                    self.payload['shared_input_gradient'] = gradient.detach().cpu().contiguous()
                    return gradient

                isolated_input.register_hook(shared_input_hook)
                self._shared_input_hook_registered = True
            self.mlp_module = module
            return args, kwargs

        return hook

    def _mlp_forward_hook(self, name):

        def hook(module, args, kwargs, output):
            if self._context() is None:
                return
            leaves = self._named_tensor_leaves(output)
            if not leaves:
                raise RuntimeError(f'MLP merge target returned no tensor output: {name}')
            self.payload['forward_outputs'] = {
                path: tensor.detach().cpu().contiguous() for path, tensor in leaves
            }
            differentiable = {path: tensor for path, tensor in leaves if tensor.requires_grad}
            if not differentiable:
                return
            if self.mode == 'replay':
                common_gradients = self._load_source(self.dout_file).get('output_gradients', {})
                if set(common_gradients) != set(differentiable):
                    raise ValueError(
                        f'MLP output gradient paths differ: source={sorted(common_gradients)}, '
                        f'runtime={sorted(differentiable)}.')
            else:
                common_gradients = None

            for output_path, tensor in differentiable.items():

                def output_hook(gradient, path=output_path):
                    if self.mode == 'capture':
                        replacement = gradient
                    else:
                        common_gradient = common_gradients[path]
                        if tuple(common_gradient.shape) != tuple(gradient.shape):
                            raise ValueError(
                                f'MLP dout {path} shape mismatch: source={tuple(common_gradient.shape)}, '
                                f'runtime={tuple(gradient.shape)}.')
                        replacement = common_gradient.to(
                            device=gradient.device, dtype=gradient.dtype)
                    self.payload['output_gradients'][path] = (
                        replacement.detach().cpu().contiguous())
                    self._seen_output_gradients.add(path)
                    return replacement

                tensor.register_hook(output_hook)

        return hook

    def _fc1_backward_hook(self, name):

        def hook(module, grad_input, grad_output):
            if self._context() is None or self.payload is None:
                return
            gradient = self._first_tensor(grad_input)
            if gradient is None:
                raise RuntimeError(f'MLP FC1 target returned no local input gradient: {name}')
            self.payload['mlp_local_input_gradient'] = gradient.detach().cpu().contiguous()
            self.payload['mlp_local_input_gradient_source'] = 'fc1_module_full_backward_hook'

        return hook

    def _layer_backward_hook(self, name):

        def hook(module, grad_input, grad_output):
            if self._context() is None or self.payload is None:
                return
            gradient = self._first_tensor(grad_output)
            if gradient is None:
                raise RuntimeError(f'MLP merge layer returned no output gradient: {name}')
            self.payload['layer_output_gradient'] = gradient.detach().cpu().contiguous()

        return hook

    def register_hooks(self):
        matched = {'layer': [], 'mlp': [], 'fc1': []}
        for model in self.trainer.unwrapped_models:
            for name, module in model.named_modules():
                if name == self.layer_target:
                    self.handles.append(module.register_full_backward_hook(
                        self._layer_backward_hook(name)))
                    matched['layer'].append(f'{name} ({type(module).__name__})')
                elif name == self.mlp_target:
                    self.handles.extend([
                        module.register_forward_pre_hook(
                            self._mlp_pre_hook(name), with_kwargs=True),
                        module.register_forward_hook(
                            self._mlp_forward_hook(name), with_kwargs=True),
                    ])
                    matched['mlp'].append(f'{name} ({type(module).__name__})')
                elif name == self.fc1_target:
                    self.handles.append(module.register_full_backward_hook(
                        self._fc1_backward_hook(name)))
                    matched['fc1'].append(f'{name} ({type(module).__name__})')
        missing = [name for name, values in matched.items() if not values]
        if missing:
            raise ValueError(
                f'MLP merge isolation targets not found: {missing}; matched={matched}.')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f'GKD MLP merge isolation mode={self.mode}, output={self.output_path}, '
            f'matched={matched}')

    def prepare_train_step(self, step, debug_active, num_microbatches):
        if not self.enabled or step != self.step:
            return
        if num_microbatches != 1:
            raise ValueError(
                'MLP merge isolation requires exactly one micro-batch. '
                'Set global_batch_size equal to micro_batch_size.')
        if not debug_active:
            raise ValueError('MLP merge isolation step must be inside the alignment debug window.')

    def finalize_train_step(self, step):
        if not self.enabled or step != self.step:
            return
        if self.payload is None:
            raise RuntimeError('MLP merge isolation target did not run.')
        if not self._seen_output_gradients:
            raise RuntimeError('MLP merge isolation observed no MLP output gradient.')
        if self.payload.get('mlp_local_input_gradient') is None:
            raise RuntimeError('MLP merge isolation did not capture the local MLP input gradient.')
        if self.mode == 'capture':
            if self.payload.get('layer_output_gradient') is None:
                raise RuntimeError('MLP merge capture did not observe the layer output gradient.')
            if self.payload.get('shared_input_gradient') is None:
                raise RuntimeError(
                    'MLP merge capture did not observe the shared residual input gradient. '
                    'Check activation-checkpoint boundaries and the selected layer.')
        torch.save(self.payload, self.output_path)
        logger.info(f'Saved GKD MLP merge isolation result: {self.output_path}')
