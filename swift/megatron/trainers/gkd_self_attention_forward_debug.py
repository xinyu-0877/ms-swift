# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import inspect
import os

import torch

from swift.utils import get_logger


logger = get_logger()


class GKDSelfAttentionForwardIsolation:
    """Replay a complete self-attention branch with common tensor inputs and parameters."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.mode = os.getenv('SWIFT_GKD_SELF_ATTENTION_FORWARD_MODE', '').lower()
        self.output_dir = os.getenv('SWIFT_GKD_SELF_ATTENTION_FORWARD_DIR')
        self.layer_target = os.getenv(
            'SWIFT_GKD_SELF_ATTENTION_FORWARD_LAYER_TARGET', 'decoder.layers.0')
        self.attention_target = f'{self.layer_target}.self_attention'
        self.mlp_target = f'{self.layer_target}.mlp'
        self.step = int(os.getenv('SWIFT_GKD_SELF_ATTENTION_FORWARD_STEP', '0'))
        self.micro_batch = int(os.getenv('SWIFT_GKD_SELF_ATTENTION_FORWARD_MICRO_BATCH', '0'))
        self.tag = os.getenv('SWIFT_GKD_SELF_ATTENTION_FORWARD_TAG', '').strip()
        self.source_file = os.getenv('SWIFT_GKD_SELF_ATTENTION_FORWARD_SOURCE')
        self.handles = []
        self.payload = None
        self.source = None
        self._attention_seen = False
        self._post_attention_seen = False

        if self.mode not in {'', 'capture', 'replay'}:
            raise ValueError(
                'SWIFT_GKD_SELF_ATTENTION_FORWARD_MODE must be empty, "capture", or "replay".')
        if not self.mode:
            return
        if not self.output_dir:
            raise ValueError('SWIFT_GKD_SELF_ATTENTION_FORWARD_DIR is required.')
        if self.mode == 'replay' and not self.source_file:
            raise ValueError('SWIFT_GKD_SELF_ATTENTION_FORWARD_SOURCE is required for replay.')
        if self.step < 0 or self.micro_batch < 0:
            raise ValueError('Self-attention forward step and micro-batch must be non-negative.')
        if not self.tag or not self.tag.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Self-attention forward tag contains unsupported characters.')

    @property
    def enabled(self):
        return bool(self.mode)

    @property
    def output_path(self):
        return os.path.join(self.output_dir, f'self_attention_forward_{self.tag}.pt')

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

    @staticmethod
    def _tensor_leaves(value, path=()):
        if torch.is_tensor(value):
            return [(path, value)]
        result = []
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                result.extend(GKDSelfAttentionForwardIsolation._tensor_leaves(
                    item, path + (index,)))
        elif isinstance(value, dict):
            for key, item in value.items():
                result.extend(GKDSelfAttentionForwardIsolation._tensor_leaves(
                    item, path + (key,)))
        return result

    @staticmethod
    def _replace_path(value, path, replacement):
        if not path:
            return replacement
        key, *remaining = path
        if isinstance(value, tuple):
            items = list(value)
            items[key] = GKDSelfAttentionForwardIsolation._replace_path(
                items[key], tuple(remaining), replacement)
            return tuple(items)
        if isinstance(value, list):
            items = list(value)
            items[key] = GKDSelfAttentionForwardIsolation._replace_path(
                items[key], tuple(remaining), replacement)
            return items
        if isinstance(value, dict):
            items = dict(value)
            items[key] = GKDSelfAttentionForwardIsolation._replace_path(
                items[key], tuple(remaining), replacement)
            return items
        raise ValueError(f'Cannot replace tensor at path {path}; parent type={type(value).__name__}.')

    @staticmethod
    def _path_key(root, path):
        return repr((root,) + tuple(path))

    @staticmethod
    def _first_tensor(value):
        leaves = GKDSelfAttentionForwardIsolation._tensor_leaves(value)
        return leaves[0][1] if leaves else None

    def _context_matches(self):
        context = self.trainer._operator_debug_context
        return (context is not None and context['step'] == self.step
                and context['micro_batch'] == self.micro_batch)

    def _load_source(self):
        if self.source is None:
            if not os.path.isfile(self.source_file):
                raise FileNotFoundError(
                    f'Self-attention replay source not found: {self.source_file}')
            self.source = torch.load(self.source_file, map_location='cpu', weights_only=True)
            if self.source.get('mode') != 'capture':
                raise ValueError('Self-attention replay source must be a capture payload.')
            if self.source.get('attention_target') != self.attention_target:
                raise ValueError(
                    f'Self-attention source target={self.source.get("attention_target")}, '
                    f'expected={self.attention_target}.')
        return self.source

    def _runtime_config(self, module):
        args = self.trainer.args
        return {
            'module_training': bool(module.training),
            'hidden_dropout': float(getattr(args, 'hidden_dropout', 0.0) or 0.0),
            'attention_dropout': float(getattr(args, 'attention_dropout', 0.0) or 0.0),
            'padding_free': bool(getattr(args, 'padding_free', False)),
            'sequence_parallel': bool(getattr(args, 'sequence_parallel', False)),
            'tensor_model_parallel_size': int(getattr(args, 'tensor_model_parallel_size', 1)),
            'pipeline_model_parallel_size': int(getattr(args, 'pipeline_model_parallel_size', 1)),
            'context_parallel_size': int(getattr(args, 'context_parallel_size', 1)),
        }

    def _attention_pre_hook(self, name):

        def hook(module, args, kwargs):
            if not self._context_matches() or self._attention_seen:
                return args, kwargs
            runtime_parameters = dict(module.named_parameters())
            runtime_leaves = []
            runtime_leaves.extend(
                ('args', path, tensor) for path, tensor in self._tensor_leaves(args))
            runtime_leaves.extend(
                ('kwargs', path, tensor) for path, tensor in self._tensor_leaves(kwargs))
            if not runtime_leaves:
                raise RuntimeError(f'Self-attention target received no tensor input: {name}')
            runtime_by_key = {
                self._path_key(root, path): (root, path, tensor)
                for root, path, tensor in runtime_leaves
            }
            runtime_config = self._runtime_config(module)
            if runtime_config['hidden_dropout'] != 0.0 or runtime_config['attention_dropout'] != 0.0:
                raise ValueError(
                    'Self-attention common-input replay requires hidden_dropout=0 and '
                    'attention_dropout=0 unless common RNG/dropout masks are implemented; '
                    f'got {runtime_config}.')

            if self.mode == 'capture':
                isolated_args, isolated_kwargs = args, kwargs
            else:
                source = self._load_source()
                if source.get('runtime_config') != runtime_config:
                    raise ValueError(
                        f'Self-attention runtime config mismatch: '
                        f'source={source.get("runtime_config")}, runtime={runtime_config}.')
                common_inputs = source.get('inputs', {})
                if set(common_inputs) != set(runtime_by_key):
                    raise ValueError(
                        f'Self-attention tensor input paths differ: '
                        f'source={sorted(common_inputs)}, runtime={sorted(runtime_by_key)}.')
                isolated_args, isolated_kwargs = args, kwargs
                for key, common in common_inputs.items():
                    root, path, runtime = runtime_by_key[key]
                    if tuple(common.shape) != tuple(runtime.shape):
                        raise ValueError(
                            f'Self-attention input {key} shape mismatch: '
                            f'source={tuple(common.shape)}, runtime={tuple(runtime.shape)}.')
                    replacement = common.to(
                        device=runtime.device, dtype=runtime.dtype).detach()
                    if root == 'args':
                        isolated_args = self._replace_path(isolated_args, path, replacement)
                    else:
                        isolated_kwargs = self._replace_path(isolated_kwargs, path, replacement)

                common_parameters = source.get('parameters', {})
                if set(common_parameters) != set(runtime_parameters):
                    raise ValueError(
                        f'Self-attention parameter names differ: '
                        f'source={sorted(common_parameters)}, runtime={sorted(runtime_parameters)}.')
                with torch.no_grad():
                    for parameter_name, parameter in runtime_parameters.items():
                        common = common_parameters[parameter_name]
                        if tuple(common.shape) != tuple(parameter.shape):
                            raise ValueError(
                                f'Self-attention parameter {parameter_name} shape mismatch: '
                                f'source={tuple(common.shape)}, runtime={tuple(parameter.shape)}.')
                        parameter.copy_(common.to(device=parameter.device, dtype=parameter.dtype))

            actual_leaves = []
            actual_leaves.extend(
                ('args', path, tensor) for path, tensor in self._tensor_leaves(isolated_args))
            actual_leaves.extend(
                ('kwargs', path, tensor) for path, tensor in self._tensor_leaves(isolated_kwargs))
            self.payload = {
                'mode': self.mode,
                'tag': self.tag,
                'step': self.step,
                'micro_batch': self.micro_batch,
                'layer_target': self.layer_target,
                'attention_target': self.attention_target,
                'mlp_target': self.mlp_target,
                'module_type': type(module).__name__,
                'module_signature': str(inspect.signature(module.forward)),
                'runtime_config': runtime_config,
                'inputs': {
                    self._path_key(root, path): tensor.detach().cpu().contiguous()
                    for root, path, tensor in actual_leaves
                },
                'input_identities': {
                    self._path_key(root, path): self._identity(tensor)
                    for root, path, tensor in actual_leaves
                },
                'parameters': {
                    parameter_name: parameter.detach().cpu().contiguous()
                    for parameter_name, parameter in runtime_parameters.items()
                },
                'parameter_identities': {
                    parameter_name: self._identity(parameter)
                    for parameter_name, parameter in runtime_parameters.items()
                },
                'attention_outputs': {},
                'post_attention_input': None,
                'post_attention_input_identity': None,
            }
            self._attention_seen = True
            return isolated_args, isolated_kwargs

        return hook

    def _attention_forward_hook(self, name):

        def hook(module, args, kwargs, output):
            if not self._context_matches() or self.payload is None:
                return
            leaves = self._tensor_leaves(output)
            if not leaves:
                raise RuntimeError(f'Self-attention target returned no tensor output: {name}')
            self.payload['attention_outputs'] = {
                repr(path): tensor.detach().cpu().contiguous()
                for path, tensor in leaves
            }

        return hook

    def _mlp_pre_hook(self, name):

        def hook(module, args, kwargs):
            if not self._context_matches() or self.payload is None or self._post_attention_seen:
                return args, kwargs
            tensor = self._first_tensor(kwargs)
            if tensor is None:
                tensor = self._first_tensor(args)
            if tensor is None:
                raise RuntimeError(f'MLP target received no post-attention tensor: {name}')
            self.payload['post_attention_input'] = tensor.detach().cpu().contiguous()
            self.payload['post_attention_input_identity'] = self._identity(tensor)
            self._post_attention_seen = True
            return args, kwargs

        return hook

    def register_hooks(self):
        matched = {'attention': [], 'mlp': []}
        for model in self.trainer.unwrapped_models:
            for name, module in model.named_modules():
                if name == self.attention_target:
                    self.handles.extend([
                        module.register_forward_pre_hook(
                            self._attention_pre_hook(name), with_kwargs=True),
                        module.register_forward_hook(
                            self._attention_forward_hook(name), with_kwargs=True),
                    ])
                    matched['attention'].append(f'{name} ({type(module).__name__})')
                elif name == self.mlp_target:
                    self.handles.append(module.register_forward_pre_hook(
                        self._mlp_pre_hook(name), with_kwargs=True))
                    matched['mlp'].append(f'{name} ({type(module).__name__})')
        if len(matched['attention']) != 1 or len(matched['mlp']) != 1:
            raise ValueError(
                f'Expected exactly one self-attention and MLP target; matched={matched}.')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f'GKD self-attention forward isolation mode={self.mode}, '
            f'output={self.output_path}, matched={matched}')

    def prepare_train_step(self, step, debug_active, num_microbatches):
        if not self.enabled or step != self.step:
            return
        if num_microbatches != 1:
            raise ValueError(
                'Self-attention forward isolation requires exactly one micro-batch. '
                'Set global_batch_size equal to micro_batch_size.')
        if not debug_active:
            raise ValueError(
                'Self-attention forward isolation step must be inside the alignment debug window.')
        parallel_sizes = {
            name: int(getattr(self.trainer.args, name, 1))
            for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size',
                         'context_parallel_size')
        }
        invalid = {name: size for name, size in parallel_sizes.items() if size != 1}
        if invalid:
            raise ValueError(f'Self-attention full-tensor isolation requires TP=PP=CP=1; got {invalid}.')

    def finalize_train_step(self, step):
        if not self.enabled or step != self.step:
            return
        if self.payload is None or not self._attention_seen:
            raise RuntimeError('Self-attention isolation target did not run.')
        if not self.payload.get('attention_outputs'):
            raise RuntimeError('Self-attention isolation captured no branch output.')
        if not self._post_attention_seen:
            raise RuntimeError('Self-attention isolation captured no post-attention MLP input.')
        torch.save(self.payload, self.output_path)
        logger.info(f'Saved GKD self-attention forward isolation: {self.output_path}')
