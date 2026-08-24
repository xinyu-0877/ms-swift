# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import re

import torch

from swift.utils import get_logger


logger = get_logger()


class GKDAttentionForwardTrace:
    """Capture decoder-layer outputs or Layer-N boundaries, including core-attention Q/K/V."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.enabled = os.getenv(
            'SWIFT_GKD_ATTENTION_FORWARD_TRACE', '0').lower() in {'1', 'true', 'yes'}
        self.output_dir = os.getenv('SWIFT_GKD_ATTENTION_FORWARD_DIR')
        self.scope = os.getenv(
            'SWIFT_GKD_ATTENTION_FORWARD_SCOPE', 'layer').lower()
        self.layer_target = os.getenv(
            'SWIFT_GKD_ATTENTION_FORWARD_LAYER_TARGET', 'decoder.layers.27')
        self.step = int(os.getenv('SWIFT_GKD_ATTENTION_FORWARD_STEP', '0'))
        self.micro_batch = int(os.getenv('SWIFT_GKD_ATTENTION_FORWARD_MICRO_BATCH', '0'))
        self.tag = os.getenv('SWIFT_GKD_ATTENTION_FORWARD_TAG', 'capture').lower()
        self.handles = []
        self.payload = None
        self._captured_nodes = set()

        self.node_targets = {}
        if self.scope == 'layer':
            self.node_targets = {
                'A_layer_input': self.layer_target,
                'B0_linear_qkv_output': f'{self.layer_target}.self_attention.linear_qkv',
                'B1_query_before_qk_norm': f'{self.layer_target}.self_attention.q_layernorm',
                'B1_key_before_qk_norm': f'{self.layer_target}.self_attention.k_layernorm',
                'B2_query_after_qk_norm': f'{self.layer_target}.self_attention.q_layernorm',
                'B2_key_after_qk_norm': f'{self.layer_target}.self_attention.k_layernorm',
                'C0_core_attention_input_qkv': f'{self.layer_target}.self_attention.core_attention',
                'C_core_attention_output': f'{self.layer_target}.self_attention.core_attention',
                'D_linear_proj_output': f'{self.layer_target}.self_attention.linear_proj',
                'E_pre_mlp_input': f'{self.layer_target}.mlp',
                'F_mlp_output': f'{self.layer_target}.mlp',
                'G_layer_output': self.layer_target,
            }
        if not self.enabled:
            return
        if not self.output_dir:
            raise ValueError(
                'SWIFT_GKD_ATTENTION_FORWARD_DIR is required when attention forward trace is enabled.')
        if self.scope not in {'layer', 'layers'}:
            raise ValueError(
                'SWIFT_GKD_ATTENTION_FORWARD_SCOPE must be "layer" or "layers".')
        if self.step < 0 or self.micro_batch < 0:
            raise ValueError('Attention forward trace step and micro-batch must be non-negative.')
        if not self.tag or not self.tag.replace('_', '').replace('-', '').isalnum():
            raise ValueError(
                'SWIFT_GKD_ATTENTION_FORWARD_TAG must contain only letters, digits, underscores, or hyphens.')

    @property
    def output_path(self):
        return os.path.join(self.output_dir, f'attention_forward_{self.tag}.pt')

    def _context_matches(self):
        context = self.trainer._operator_debug_context
        return (
            context is not None
            and context['step'] == self.step
            and context['micro_batch'] == self.micro_batch
        )

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
            tensor = GKDAttentionForwardTrace._first_tensor(item)
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
                result.extend(GKDAttentionForwardTrace._named_tensor_leaves(
                    item, f'{prefix}.{key}'))
            return result
        if isinstance(value, (tuple, list)):
            result = []
            for index, item in enumerate(value):
                result.extend(GKDAttentionForwardTrace._named_tensor_leaves(
                    item, f'{prefix}.{index}'))
            return result
        return []

    @staticmethod
    def _module_input(args, kwargs):
        for key in ('hidden_states', 'input_tensor', 'x'):
            value = kwargs.get(key)
            if torch.is_tensor(value):
                return value
        tensor = GKDAttentionForwardTrace._first_tensor(args)
        if tensor is None:
            tensor = GKDAttentionForwardTrace._first_tensor(kwargs)
        return tensor

    def _initialize_payload(self):
        if self.payload is None:
            self.payload = {
                'tag': self.tag,
                'step': self.step,
                'micro_batch': self.micro_batch,
                'scope': self.scope,
                'layer_target': self.layer_target,
                'node_targets': dict(self.node_targets),
                'module_types': {},
                'nodes': {},
            }

    def _capture(self, node, module, values):
        if node in self._captured_nodes or not self._context_matches():
            return
        leaves = self._named_tensor_leaves(values, prefix='tensor')
        if not leaves:
            raise RuntimeError(f'Attention forward trace node {node} produced no tensor.')
        self._initialize_payload()
        self.payload['module_types'][node] = type(module).__name__
        self.payload['nodes'][node] = {
            path: tensor.detach().cpu().contiguous()
            for path, tensor in leaves
        }
        self._captured_nodes.add(node)

    def _pre_hook(self, node):

        def hook(module, args, kwargs):
            if not self._context_matches() or node in self._captured_nodes:
                return args, kwargs
            input_tensor = self._module_input(args, kwargs)
            if input_tensor is None:
                raise RuntimeError(f'Attention forward trace node {node} received no tensor input.')
            self._capture(node, module, input_tensor)
            return args, kwargs

        return hook

    @staticmethod
    def _core_attention_qkv(module, args, kwargs):
        arguments = dict(kwargs)
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
            arguments.update(bound)
            if isinstance(bound.get('kwargs'), dict):
                arguments.update(bound['kwargs'])
        except (TypeError, ValueError):
            pass
        aliases = {
            'query': ('query', 'query_layer', 'q'),
            'key': ('key', 'key_layer', 'k'),
            'value': ('value', 'value_layer', 'v'),
        }
        qkv = {}
        for output_name, names in aliases.items():
            for name in names:
                value = arguments.get(name)
                if torch.is_tensor(value):
                    qkv[output_name] = value
                    break
        if len(qkv) == 3:
            return qkv

        positional_tensors = [value for value in args if torch.is_tensor(value)]
        if len(positional_tensors) >= 3:
            return dict(zip(('query', 'key', 'value'), positional_tensors[:3]))
        raise RuntimeError(
            'Attention forward trace could not identify query/key/value at the core-attention input: '
            f'signature={inspect.signature(module.forward)}, argument_names={list(arguments)}')

    def _core_attention_pre_hook(self, node):

        def hook(module, args, kwargs):
            if not self._context_matches() or node in self._captured_nodes:
                return args, kwargs
            self._capture(node, module, self._core_attention_qkv(module, args, kwargs))
            return args, kwargs

        return hook

    def _forward_hook(self, node):

        def hook(module, args, kwargs, output):
            self._capture(node, module, output)

        return hook

    def register_hooks(self):
        if self.scope == 'layers':
            return self._register_layer_scan_hooks()
        matched = {node: [] for node in self.node_targets}
        target_to_nodes = {}
        for node, target in self.node_targets.items():
            target_to_nodes.setdefault(target, []).append(node)
        for model in self.trainer.unwrapped_models:
            for name, module in model.named_modules():
                nodes = target_to_nodes.get(name)
                if nodes is None:
                    continue
                for node in nodes:
                    if node == 'C0_core_attention_input_qkv':
                        handle = module.register_forward_pre_hook(
                            self._core_attention_pre_hook(node), with_kwargs=True)
                    elif node in {
                            'A_layer_input',
                            'B1_query_before_qk_norm',
                            'B1_key_before_qk_norm',
                            'E_pre_mlp_input',
                    }:
                        handle = module.register_forward_pre_hook(
                            self._pre_hook(node), with_kwargs=True)
                    else:
                        handle = module.register_forward_hook(
                            self._forward_hook(node), with_kwargs=True)
                    self.handles.append(handle)
                    matched[node].append(f'{name} ({type(module).__name__})')
        missing = [node for node, modules in matched.items() if not modules]
        if missing:
            raise ValueError(
                f'Attention forward trace targets not found: {missing}; matched={matched}')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f'GKD attention forward trace enabled: layer={self.layer_target}, '
            f'output={self.output_path}, matched={matched}')

    def _register_layer_scan_hooks(self):
        matched = []
        for model in self.trainer.unwrapped_models:
            for name, module in model.named_modules():
                match = re.fullmatch(r'decoder\.layers\.(\d+)', name)
                if match is None:
                    continue
                layer_index = int(match.group(1))
                input_node = f'L{layer_index:02d}_input'
                output_node = f'L{layer_index:02d}_output'
                self.node_targets[input_node] = name
                self.node_targets[output_node] = name
                self.handles.extend([
                    module.register_forward_pre_hook(
                        self._pre_hook(input_node), with_kwargs=True),
                    module.register_forward_hook(
                        self._forward_hook(output_node), with_kwargs=True),
                ])
                matched.append(f'{name} ({type(module).__name__})')
        if not matched:
            raise ValueError('Attention forward layer scan found no decoder.layers.N modules.')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f'GKD full layer forward scan enabled: layers={len(matched)}, '
            f'output={self.output_path}, matched={matched}')

    def prepare_train_step(self, step, debug_active, num_microbatches):
        if not self.enabled or step != self.step:
            return
        if num_microbatches != 1:
            raise ValueError(
                'Attention forward trace requires exactly one micro-batch. '
                'Set global_batch_size equal to micro_batch_size.')
        if not debug_active:
            raise ValueError('Attention forward trace step must be inside the alignment debug window.')

    def finalize_train_step(self, step):
        if not self.enabled or step != self.step:
            return
        expected = set(self.node_targets)
        missing = sorted(expected - self._captured_nodes)
        if missing:
            raise RuntimeError(f'Attention forward trace did not capture nodes: {missing}')
        torch.save(self.payload, self.output_path)
        logger.info(f'Saved GKD attention forward trace: {self.output_path}')
