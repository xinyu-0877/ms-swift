# Copyright (c) ModelScope Contributors. All rights reserved.
"""Step-0 Layer-0 A-G forward capture for ordinary Megatron SFT runs.

This deliberately does not depend on GKD loss or teacher logits.  It reuses
the cross-backend hook implementation, but records only the tensors needed to
compare a single decoder layer under identical input provenance.
"""

import hashlib
import os

import torch

from swift.utils import get_logger
from .gkd_attention_forward_debug import GKDAttentionForwardTrace

logger = get_logger()


class SFTAttentionForwardTrace(GKDAttentionForwardTrace):
    """Capture full-tensor A-G boundaries from a normal SFT forward pass."""

    A_G_NODES = (
        'A_layer_input',
        'B0_linear_qkv_output',
        'C_core_attention_output',
        'D_linear_proj_output',
        'E_pre_mlp_input',
        'F_mlp_output',
        'G_layer_output',
    )

    def __init__(self, trainer):
        # Reuse the parent's hook methods without calling its constructor:
        # a process may run GKD diagnostics with SWIFT_GKD_* variables set,
        # and those variables must not configure this SFT tracer.
        self.trainer = trainer
        self.handles = []
        self.payload = None
        self._captured_nodes = set()
        self.enabled = os.getenv(
            'SWIFT_SFT_ATTENTION_FORWARD_TRACE', '0').lower() in {'1', 'true', 'yes'}
        self.output_dir = os.getenv('SWIFT_SFT_ATTENTION_FORWARD_DIR')
        self.scope = 'layer'
        self.layer_target = os.getenv(
            'SWIFT_SFT_ATTENTION_FORWARD_LAYER_TARGET', 'decoder.layers.0')
        self.step = int(os.getenv('SWIFT_SFT_ATTENTION_FORWARD_STEP', '0'))
        self.micro_batch = int(os.getenv('SWIFT_SFT_ATTENTION_FORWARD_MICRO_BATCH', '0'))
        self.tag = os.getenv('SWIFT_SFT_ATTENTION_FORWARD_TAG', 'capture').lower()
        self.node_targets = {
            'A_layer_input': self.layer_target,
            'B0_linear_qkv_output': f'{self.layer_target}.self_attention.linear_qkv',
            'C_core_attention_output': f'{self.layer_target}.self_attention.core_attention',
            'D_linear_proj_output': f'{self.layer_target}.self_attention.linear_proj',
            'E_pre_mlp_input': f'{self.layer_target}.mlp',
            'F_mlp_output': f'{self.layer_target}.mlp',
            'G_layer_output': self.layer_target,
        }
        self._trace_context = None
        if not self.enabled:
            return
        if not self.output_dir:
            raise ValueError(
                'SWIFT_SFT_ATTENTION_FORWARD_DIR is required when SFT attention trace is enabled.')
        if self.step < 0 or self.micro_batch < 0:
            raise ValueError('SFT attention trace step and micro-batch must be non-negative.')
        if not self.tag or not self.tag.replace('_', '').replace('-', '').isalnum():
            raise ValueError(
                'SWIFT_SFT_ATTENTION_FORWARD_TAG must contain only letters, digits, underscores, or hyphens.')
        self.parameter_patterns = [
            value.strip() for value in os.getenv(
                'SWIFT_SFT_ATTENTION_FORWARD_PARAMETER_PATTERNS',
                'embedding.word_embeddings.weight,decoder.layers.0.self_attention.linear_qkv.weight,'
                'decoder.layers.0.mlp.linear_fc2.weight').split(',') if value.strip()
        ]

    @property
    def output_path(self):
        return os.path.join(self.output_dir, f'sft_attention_forward_{self.tag}.pt')

    @staticmethod
    def _is_debug_rank():
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    @staticmethod
    def _tensor_identity(tensor):
        if tensor is None:
            return None
        value = tensor.detach().cpu().contiguous()
        raw = value.numpy().tobytes()
        return {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'numel': value.numel(),
            'sha256': hashlib.sha256(raw).hexdigest(),
        }

    def _parameter_probes(self):
        result = []
        for model in self.trainer.unwrapped_models:
            for name, parameter in model.named_parameters():
                if not any(name == pattern or name.endswith(pattern) for pattern in self.parameter_patterns):
                    continue
                value = parameter.detach().float().cpu().contiguous()
                flat = value.reshape(-1)
                sample_count = min(256, flat.numel())
                indices = torch.linspace(0, flat.numel() - 1, sample_count, dtype=torch.long)
                sample = flat[indices]
                result.append({
                    'name': name,
                    'shape': list(value.shape),
                    'dtype': str(parameter.dtype),
                    'numel': value.numel(),
                    'sample_indices': indices.tolist(),
                    'sample_sha256_fp32': hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
                })
        return result

    def _context_matches(self):
        return (
            self._trace_context is not None
            and self._trace_context['step'] == self.step
            and self._trace_context['micro_batch'] == self.micro_batch
        )

    def _initialize_payload(self):
        if self.payload is not None:
            return
        args = self.trainer.args
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        model_parallel_size = (
            int(getattr(args, 'tensor_model_parallel_size', 1))
            * int(getattr(args, 'pipeline_model_parallel_size', 1))
            * int(getattr(args, 'context_parallel_size', 1)))
        runtime_keys = (
            'tensor_model_parallel_size', 'pipeline_model_parallel_size',
            'context_parallel_size', 'micro_batch_size', 'global_batch_size',
            'padding_free', 'sequence_parallel', 'attention_backend', 'torch_dtype',
        )
        runtime = {key: str(getattr(args, key, None)) for key in runtime_keys}
        runtime.update({
            'world_size': world_size,
            'data_parallel_size': world_size // model_parallel_size,
            'fp32_residual_connection': bool(
                getattr(self.trainer.config, 'fp32_residual_connection', False)),
        })
        self.payload = {
            'record_type': 'sft_attention_forward',
            'tag': self.tag,
            'step': self.step,
            'micro_batch': self.micro_batch,
            'scope': self.scope,
            'layer_target': self.layer_target,
            'node_targets': dict(self.node_targets),
            'module_types': {},
            'nodes': {},
            'runtime': runtime,
            'checkpoint_provenance': {
                key: str(getattr(args, key, None))
                for key in ('model', 'load', 'finetune', 'seed', 'data_seed')
            },
        }

    def capture_provenance(self, data, labels, step, micro_batch):
        if (not self.enabled or not self._is_debug_rank()
                or self._trace_context is None
                or step != self.step or micro_batch != self.micro_batch):
            return
        self._initialize_payload()
        if 'provenance' in self.payload:
            raise RuntimeError('SFT attention trace provenance captured more than once.')
        self.payload['provenance'] = {
            'input_ids': self._tensor_identity(data.get('input_ids')),
            'position_ids': self._tensor_identity(data.get('position_ids')),
            'labels': self._tensor_identity(labels),
            'num_valid': int((labels != -100).sum().item()) if labels is not None else None,
            'student_parameter_probes': self._parameter_probes(),
        }

    def prepare_train_step(self, step, num_microbatches):
        if not self.enabled or step != self.step:
            return
        if num_microbatches != 1:
            raise ValueError(
                'SFT attention trace requires exactly one micro-batch. '
                'Set global_batch_size equal to micro_batch_size.')
        invalid = {
            name: int(getattr(self.trainer.args, name, 1))
            for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size',
                         'context_parallel_size')
            if int(getattr(self.trainer.args, name, 1)) != 1
        }
        if invalid:
            raise ValueError(
                'SFT full-tensor trace requires TP=1, PP=1, and CP=1; '
                f'got {invalid}.')
        if not self._is_debug_rank():
            return
        self._trace_context = {'step': step, 'micro_batch': self.micro_batch}
        self.payload = None
        self._captured_nodes.clear()

    def finalize_train_step(self, step):
        if not self.enabled or step != self.step or not self._is_debug_rank():
            return
        if self.payload is None or 'provenance' not in self.payload:
            raise RuntimeError('SFT attention trace did not capture input provenance.')
        missing = sorted(set(self.node_targets) - self._captured_nodes)
        if missing:
            raise RuntimeError(f'SFT attention trace did not capture nodes: {missing}')
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.payload, self.output_path)
        logger.info(f'Saved SFT attention forward trace: {self.output_path}')
        self._trace_context = None
