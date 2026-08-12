# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import json
import os
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from functools import partial
from mcore_bridge import set_random_seed
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from transformers import AutoConfig
from transformers.utils import ContextManagers
from typing import Dict, List, Optional

from swift.infer_engine.protocol import RequestConfig
from swift.megatron.arguments import MegatronArguments
from swift.megatron.model import get_mcore_model
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, build_opsd_teacher_data, gkd_loss
from swift.rlhf_trainers.utils import (assemble_teacher_topk_logprobs, build_teacher_infer_request,
                                       get_non_thinking_prefix_ids, parse_prompt_logprobs,
                                       replace_assistant_response_with_ids)
from swift.rlhf_trainers.vllm_client import VLLMInferClient
from swift.template import Template
from swift.utils import get_cu_seqlens_from_position_ids, get_logger, is_last_rank, to_device
from ..utils import forward_step_helper, get_padding_to
from .gkd_utils import cp_reduce, tp_gather_topk, vocab_parallel_topk
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from .vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax

logger = get_logger()


class MegatronGKDTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    def __init__(self, args: MegatronArguments, template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client', None)

        # GKD-specific parameters
        self.beta = args.beta  # JSD interpolation coefficient
        self.temperature = args.temperature
        self.lmbda = args.lmbda  # On-policy probability
        self.seq_kd = args.seq_kd  # Sequential KD: use teacher-generated responses
        self.offload_teacher_model = args.offload_teacher_model  # Offload teacher to CPU
        self.teacher_model_server = getattr(args, 'teacher_model_server', None)
        self.use_teacher_api = self.teacher_model_server is not None
        self._is_self_distillation = (args.teacher_model is None and self.teacher_model_server is None)
        self._teacher_use_disable_adapter = getattr(args, '_teacher_use_disable_adapter', False)
        if self._teacher_use_disable_adapter:
            logger.info('Self-distillation mode: using disable_adapter() for fixed teacher (no extra model)')
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)  # Weight for SFT loss

        # GKD top-k logits configuration
        self.gkd_logits_topk = getattr(args, 'gkd_logits_topk', None)

        self.use_vllm = getattr(args, 'use_vllm', False)
        self.steps_per_generation = args.steps_per_generation
        self.generation_batch_size = args.generation_batch_size
        super().__init__(args, template)

        self._alignment_debug_steps = int(os.getenv('SWIFT_GKD_ALIGNMENT_DEBUG_STEPS', '0'))
        self._alignment_debug_dir = os.getenv('SWIFT_GKD_ALIGNMENT_DEBUG_DIR')
        if self._alignment_debug_steps > 0 and not self._alignment_debug_dir:
            self._alignment_debug_dir = os.path.join(args.output_dir, 'gkd_alignment_debug')
        self._alignment_full_param_sample_size = int(
            os.getenv('SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT',
                      os.getenv('SWIFT_GKD_ALIGNMENT_FULL_PARAM_SAMPLE_SIZE', '32')))
        if self._alignment_full_param_sample_size <= 0:
            raise ValueError('SWIFT_GKD_ALIGNMENT_SAMPLE_COUNT must be greater than 0.')
        param_patterns = os.getenv(
            'SWIFT_GKD_ALIGNMENT_KEY_PATTERNS', os.getenv('SWIFT_GKD_ALIGNMENT_FULL_PARAM_PATTERNS', ''))
        self._alignment_full_param_patterns = [p.strip() for p in param_patterns.split(',') if p.strip()]
        if not self._alignment_full_param_patterns:
            self._alignment_full_param_patterns = [
                'word_embeddings.weight',
                'decoder.layers.0.self_attention.linear_qkv.weight',
                'decoder.layers.2.mlp.linear_fc2.weight',
            ]
        self._alignment_micro_counts = {}
        self._operator_debug_enabled = os.getenv(
            'SWIFT_GKD_OPERATOR_DEBUG', '0').lower() in {'1', 'true', 'yes'}
        self._operator_debug_layer_io = os.getenv(
            'SWIFT_GKD_OPERATOR_DEBUG_LAYER_IO', '0').lower() in {'1', 'true', 'yes'}
        layer_ids = os.getenv('SWIFT_GKD_OPERATOR_DEBUG_LAYER_IDS', '')
        try:
            self._operator_debug_layer_ids = {
                int(layer_id.strip()) for layer_id in layer_ids.split(',') if layer_id.strip()
            }
        except ValueError as error:
            raise ValueError('SWIFT_GKD_OPERATOR_DEBUG_LAYER_IDS must be comma-separated integers.') from error
        operator_patterns = os.getenv('SWIFT_GKD_OPERATOR_DEBUG_PATTERNS', '')
        self._operator_debug_patterns = [
            pattern.strip() for pattern in operator_patterns.split(',') if pattern.strip()
        ]
        if not self._operator_debug_patterns:
            self._operator_debug_patterns = [
                'embedding.word_embeddings',
                'decoder.layers.0.input_layernorm',
                'decoder.layers.0.self_attention.linear_qkv',
                'decoder.layers.0.self_attention.core_attention',
                'decoder.layers.0.self_attention.linear_proj',
                'decoder.layers.0.pre_mlp_layernorm',
                'decoder.layers.0.mlp.linear_fc1',
                'decoder.layers.0.mlp.linear_fc2',
                'decoder.final_layernorm',
                'output_layer',
            ]
        self._operator_debug_context = None
        self._operator_debug_handles = []
        self._linear_proj_isolation_mode = os.getenv('SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE', '').lower()
        self._linear_proj_isolation_dir = os.getenv('SWIFT_GKD_LINEAR_PROJ_ISOLATION_DIR')
        self._linear_proj_isolation_target = os.getenv(
            'SWIFT_GKD_LINEAR_PROJ_ISOLATION_TARGET',
            'decoder.layers.0.self_attention.linear_proj')
        self._linear_proj_isolation_tag = os.getenv('SWIFT_GKD_LINEAR_PROJ_ISOLATION_TAG', '').lower()
        self._linear_proj_isolation_prefix = os.getenv(
            'SWIFT_GKD_LINEAR_PROJ_ISOLATION_PREFIX', 'linear_proj')
        if not self._linear_proj_isolation_prefix.replace('_', '').isalnum():
            raise ValueError('SWIFT_GKD_LINEAR_PROJ_ISOLATION_PREFIX must contain only letters, digits, or underscores.')
        self._linear_proj_isolation_done = False
        self._linear_proj_isolation_handles = []
        if self._linear_proj_isolation_mode not in {'', 'capture', 'replay'}:
            raise ValueError('SWIFT_GKD_LINEAR_PROJ_ISOLATION_MODE must be empty, "capture", or "replay".')
        if self._linear_proj_isolation_mode and not self._linear_proj_isolation_dir:
            raise ValueError('SWIFT_GKD_LINEAR_PROJ_ISOLATION_DIR is required for linear_proj isolation.')
        self._teacher_cache_mode = os.getenv('SWIFT_GKD_TEACHER_CACHE_MODE', '').lower()
        self._teacher_cache_dir = os.getenv('SWIFT_GKD_TEACHER_CACHE_DIR')
        reuse_step = os.getenv('SWIFT_GKD_TEACHER_CACHE_REUSE_STEP')
        self._teacher_cache_reuse_step = int(reuse_step) if reuse_step is not None else None
        if self._teacher_cache_mode not in {'', 'save', 'load'}:
            raise ValueError('SWIFT_GKD_TEACHER_CACHE_MODE must be empty, "save", or "load".')
        if self._teacher_cache_mode and not self._teacher_cache_dir:
            raise ValueError('SWIFT_GKD_TEACHER_CACHE_DIR is required when teacher cache mode is enabled.')
        if self._teacher_cache_reuse_step is not None:
            if self._teacher_cache_reuse_step < 0:
                raise ValueError('SWIFT_GKD_TEACHER_CACHE_REUSE_STEP must be non-negative.')
            if self._teacher_cache_mode != 'load':
                raise ValueError('SWIFT_GKD_TEACHER_CACHE_REUSE_STEP requires teacher cache mode "load".')
        if self._alignment_debug_steps > 0 and self._is_debug_rank():
            os.makedirs(self._alignment_debug_dir, exist_ok=True)
            logger.info(f'GKD alignment debug output: {self._alignment_debug_path}')
        if self._teacher_cache_mode and self._is_debug_rank():
            os.makedirs(self._teacher_cache_dir, exist_ok=True)
            logger.info(f'GKD teacher cache mode={self._teacher_cache_mode}, dir={self._teacher_cache_dir}')
            if self._teacher_cache_reuse_step is not None:
                logger.warning(
                    f'Reusing teacher logits from cache step {self._teacher_cache_reuse_step} for every step. '
                    'The input IDs and micro-batch order must repeat exactly.')

        if self.use_teacher_api:
            if is_last_rank():
                self.teacher_client = VLLMInferClient(base_urls=[self.teacher_model_server])
            else:
                self.teacher_client = None
            logger.info(f'Using teacher model API for logprobs, top_logprobs={self.gkd_logits_topk}')

        # Get device for data processing
        self.device = torch.cuda.current_device()

        # Initialize vLLM rollout engine if on-policy generation is enabled
        self._init_rollout_engine()

        # Truncation strategy for handling sequences that exceed max_length
        self.truncation_strategy = args.truncation_strategy
        self.max_completion_length = args.max_completion_length

        self.resample_data_iterator = None
        self._buffered_inputs = None

        if self._alignment_debug_active(0) and self.args.tuner_type == 'full':
            teacher_models = getattr(self, 'teacher_models', None)
            self._write_alignment_record({
                'record_type': 'model_parameters',
                'step': 0,
                'mode': 'full_parameter_sampled',
                'sample_size_per_parameter': self._alignment_full_param_sample_size,
                'patterns': self._alignment_full_param_patterns,
                'student': self._model_parameter_probe_summary(self.unwrapped_models),
                'teacher': self._model_parameter_probe_summary(teacher_models) if teacher_models else None,
            })
        if self._alignment_debug_active(0) and self._operator_debug_enabled:
            self._register_operator_debug_hooks()
        if self._alignment_debug_active(0) and self._linear_proj_isolation_mode:
            self._register_linear_proj_isolation_hooks()

    @property
    def _alignment_debug_path(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return os.path.join(self._alignment_debug_dir, f'rank{rank}_alignment.jsonl')

    @staticmethod
    def _is_debug_rank():
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def _alignment_debug_active(self, step=None):
        step = int(self.state.iteration) if step is None else int(step)
        return self._alignment_debug_steps > 0 and step < self._alignment_debug_steps and self._is_debug_rank()

    @staticmethod
    def _cpu_float_tensor(tensor):
        return tensor.detach().float().cpu().contiguous()

    @classmethod
    def _tensor_summary(cls, tensor, include_values=False):
        if tensor is None:
            return None
        original_dtype = str(tensor.dtype)
        value = cls._cpu_float_tensor(tensor).reshape(-1)
        summary = {
            'shape': list(tensor.shape),
            'dtype': original_dtype,
            'numel': value.numel(),
            'sha256_fp32': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
        if value.numel():
            summary.update({
                'min': value.min().item(),
                'max': value.max().item(),
                'mean': value.mean().item(),
                'std': value.std(unbiased=False).item(),
                'norm': value.norm().item(),
            })
        if include_values:
            summary['values'] = value.tolist()
        return summary

    @classmethod
    def _tensor_identity(cls, tensor):
        if tensor is None:
            return None
        value = tensor.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }

    def _logits_summary(self, logits, labels):
        if logits is None:
            return None
        active_positions = (labels != -100).nonzero(as_tuple=False) if labels is not None else None
        if active_positions is not None and active_positions.numel() > 0:
            batch_idx = int(active_positions[0, 0].item())
            seq_idx = int(active_positions[0, 1].item())
        else:
            batch_idx = seq_idx = 0
        selected = logits[batch_idx, seq_idx].detach().reshape(-1)
        vocab_size = selected.numel()
        sample_ids = [idx for idx in (0, 1, 2, 3, 10, 100, 1000, 10000, 50000, 100000) if idx < vocab_size]
        sample_indices = torch.tensor(sample_ids, device=selected.device, dtype=torch.int64)
        sample_logits = selected.index_select(0, sample_indices).float().cpu()
        result = {
            'batch_idx': batch_idx,
            'seq_idx': seq_idx,
            'token_ids': sample_ids,
            'values': sample_logits.tolist(),
            'norm': selected.float().norm().item(),
        }
        return result

    def _write_alignment_record(self, record):
        if not self._is_debug_rank():
            return
        os.makedirs(self._alignment_debug_dir, exist_ok=True)
        with open(self._alignment_debug_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

    @staticmethod
    def _first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            items = value.values()
        elif isinstance(value, (tuple, list)):
            items = value
        else:
            return None
        for item in items:
            tensor = MegatronGKDTrainer._first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    def _operator_tensor_summary(self, tensor):
        if tensor is None:
            return None
        sample = self._sample_parameter_tensor(tensor)
        return {
            'full_shape': list(tensor.shape),
            'full_dtype': str(tensor.dtype),
            'full_numel': tensor.numel(),
            'sample_indices': self._sample_parameter_indices(tensor.numel()),
            'sample': self._tensor_summary(sample, include_values=True),
        }

    def _operator_forward_hook(self, model_idx, name):

        def hook(module, inputs, output):
            context = self._operator_debug_context
            if context is None:
                return
            call_key = f'model{model_idx}.{name}'
            call_index = context['call_counts'].get(call_key, 0)
            context['call_counts'][call_key] = call_index + 1
            self._write_alignment_record({
                'record_type': 'operator_forward',
                'step': context['step'],
                'micro_batch': context['micro_batch'],
                'call_index': call_index,
                'name': call_key,
                'module_type': type(module).__name__,
                'input': self._operator_tensor_summary(self._first_tensor(inputs)),
                'output': self._operator_tensor_summary(self._first_tensor(output)),
            })

        return hook

    def _register_operator_debug_hooks(self):
        matched = []
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, module in model.named_modules():
                if self._operator_debug_layer_io:
                    # Layer-level mode records only decoder.layers.N containers.
                    is_layer = name.startswith('decoder.layers.') and name.count('.') == 2
                    if is_layer and self._operator_debug_layer_ids:
                        layer_id = int(name.rsplit('.', 1)[-1])
                        is_layer = layer_id in self._operator_debug_layer_ids
                    is_extra = name in {'embedding.word_embeddings', 'decoder.final_layernorm', 'output_layer'}
                    if not (is_layer or is_extra):
                        continue
                elif not any(pattern in name for pattern in self._operator_debug_patterns):
                    continue
                handle = module.register_forward_hook(self._operator_forward_hook(model_idx, name))
                self._operator_debug_handles.append(handle)
                matched.append(f'model{model_idx}.{name} ({type(module).__name__})')
        logger.info(f'GKD operator debug hooks registered: {matched}')

    @property
    def _linear_proj_isolation_input_path(self):
        return os.path.join(
            self._linear_proj_isolation_dir, f'{self._linear_proj_isolation_prefix}_input.pt')

    @property
    def _linear_proj_isolation_weight_path(self):
        return os.path.join(
            self._linear_proj_isolation_dir, f'{self._linear_proj_isolation_prefix}_weight.pt')

    def _linear_proj_isolation_output_path(self, tensor):
        tag = self._linear_proj_isolation_tag or tensor.device.type
        return os.path.join(
            self._linear_proj_isolation_dir,
            f'{self._linear_proj_isolation_prefix}_output_{tag}.pt')

    @staticmethod
    def _replace_first_tensor(args, kwargs, replacement):
        args = list(args)
        for index, value in enumerate(args):
            if torch.is_tensor(value):
                args[index] = replacement
                return tuple(args), kwargs
        kwargs = dict(kwargs)
        for key in ('hidden_states', 'input', 'x'):
            if torch.is_tensor(kwargs.get(key)):
                kwargs[key] = replacement
                return tuple(args), kwargs
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = replacement
                return tuple(args), kwargs
        raise ValueError('Unable to find the linear_proj input tensor in args or kwargs.')

    def _linear_proj_isolation_pre_hook(self, module, args, kwargs):
        context = self._operator_debug_context
        if context is None or context['step'] != 0 or context['micro_batch'] != 0:
            return args, kwargs
        if self._linear_proj_isolation_done:
            return args, kwargs

        input_tensor = self._first_tensor(args)
        if input_tensor is None:
            input_tensor = self._first_tensor(kwargs)
        if input_tensor is None:
            raise ValueError('Unable to capture the linear_proj input tensor.')

        os.makedirs(self._linear_proj_isolation_dir, exist_ok=True)
        if self._linear_proj_isolation_mode == 'capture':
            torch.save({
                'input': input_tensor.detach().cpu(),
                'shape': list(input_tensor.shape),
                'dtype': str(input_tensor.dtype),
                'target': self._linear_proj_isolation_target,
            }, self._linear_proj_isolation_input_path)
            weight = getattr(module, 'weight', None)
            bias = getattr(module, 'bias', None)
            if weight is not None:
                torch.save({
                    'weight': weight.detach().cpu(),
                    'bias': bias.detach().cpu() if bias is not None else None,
                    'weight_shape': list(weight.shape),
                    'weight_dtype': str(weight.dtype),
                    'target': self._linear_proj_isolation_target,
                }, self._linear_proj_isolation_weight_path)
            logger.info(f'Captured common linear_proj input: {self._linear_proj_isolation_input_path}')
            return args, kwargs

        if not os.path.isfile(self._linear_proj_isolation_input_path):
            raise FileNotFoundError(
                f'Common linear_proj input not found: {self._linear_proj_isolation_input_path}')
        payload = torch.load(self._linear_proj_isolation_input_path, map_location='cpu', weights_only=True)
        common_input = payload['input']
        if tuple(common_input.shape) != tuple(input_tensor.shape):
            raise ValueError(
                f'Common linear_proj input shape {tuple(common_input.shape)} does not match '
                f'runtime input shape {tuple(input_tensor.shape)}.')
        weight = getattr(module, 'weight', None)
        bias = getattr(module, 'bias', None)
        if weight is not None:
            if not os.path.isfile(self._linear_proj_isolation_weight_path):
                raise FileNotFoundError(
                    f'Common linear_proj weight not found: {self._linear_proj_isolation_weight_path}')
            weight_payload = torch.load(
                self._linear_proj_isolation_weight_path, map_location='cpu', weights_only=True)
            common_weight = weight_payload['weight']
            if tuple(common_weight.shape) != tuple(weight.shape):
                raise ValueError(
                    f'Common linear_proj weight shape {tuple(common_weight.shape)} does not match '
                    f'runtime weight shape {tuple(weight.shape)}.')
            with torch.no_grad():
                weight.copy_(common_weight.to(device=weight.device, dtype=weight.dtype))
                common_bias = weight_payload.get('bias')
                if bias is not None and common_bias is not None:
                    if tuple(common_bias.shape) != tuple(bias.shape):
                        raise ValueError(
                            f'Common linear_proj bias shape {tuple(common_bias.shape)} does not match '
                            f'runtime bias shape {tuple(bias.shape)}.')
                    bias.copy_(common_bias.to(device=bias.device, dtype=bias.dtype))
        common_input = common_input.to(device=input_tensor.device, dtype=input_tensor.dtype)
        logger.info(
            f'Replaying common linear_proj input and weight: {self._linear_proj_isolation_input_path}')
        return self._replace_first_tensor(args, kwargs, common_input)

    def _linear_proj_isolation_forward_hook(self, module, args, kwargs, output):
        context = self._operator_debug_context
        if context is None or context['step'] != 0 or context['micro_batch'] != 0:
            return
        if self._linear_proj_isolation_done:
            return
        output_tensor = self._first_tensor(output)
        if output_tensor is None:
            raise ValueError('Unable to capture the linear_proj output tensor.')
        output_path = self._linear_proj_isolation_output_path(output_tensor)
        torch.save({
            'output': output_tensor.detach().cpu(),
            'shape': list(output_tensor.shape),
            'dtype': str(output_tensor.dtype),
            'target': self._linear_proj_isolation_target,
            'mode': self._linear_proj_isolation_mode,
        }, output_path)
        self._linear_proj_isolation_done = True
        logger.info(f'Saved isolated linear_proj output: {output_path}')

    def _register_linear_proj_isolation_hooks(self):
        matched = []
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, module in model.named_modules():
                if name != self._linear_proj_isolation_target:
                    continue
                pre_handle = module.register_forward_pre_hook(
                    self._linear_proj_isolation_pre_hook, with_kwargs=True)
                output_handle = module.register_forward_hook(
                    self._linear_proj_isolation_forward_hook, with_kwargs=True)
                self._linear_proj_isolation_handles.extend([pre_handle, output_handle])
                matched.append(f'model{model_idx}.{name} ({type(module).__name__})')
        if not matched:
            raise ValueError(
                f'Linear projection isolation target not found: {self._linear_proj_isolation_target}')
        logger.info(
            f'GKD linear_proj isolation mode={self._linear_proj_isolation_mode}, targets={matched}')

    def _is_full_parameter_probe(self, name):
        return any(pattern in name for pattern in self._alignment_full_param_patterns)

    def _sample_parameter_tensor(self, tensor):
        value = tensor.detach()
        try:
            value = value.view(-1)
        except RuntimeError:
            value = value.reshape(-1)
        numel = value.numel()
        sample_size = min(numel, self._alignment_full_param_sample_size)
        if sample_size == 0:
            return torch.empty(0, dtype=torch.float32)
        if sample_size == numel:
            sampled = value
        elif sample_size == 1:
            sampled = value[:1]
        else:
            indices = torch.arange(sample_size, device=value.device, dtype=torch.int64)
            indices = indices * (numel - 1) // (sample_size - 1)
            sampled = value.index_select(0, indices)
        return self._cpu_float_tensor(sampled)

    def _model_parameter_probe_summary(self, models):
        total_numel = 0
        trainable_numel = 0
        probes = []
        for model_idx, model in enumerate(models or []):
            for name, parameter in model.named_parameters():
                total_numel += parameter.numel()
                if parameter.requires_grad:
                    trainable_numel += parameter.numel()
                if not self._is_full_parameter_probe(name):
                    continue
                sample = self._sample_parameter_tensor(parameter)
                probes.append({
                    'name': f'model{model_idx}.{name}',
                    'full_shape': list(parameter.shape),
                    'full_dtype': str(parameter.dtype),
                    'full_numel': parameter.numel(),
                    'sample_indices': self._sample_parameter_indices(parameter.numel()),
                    'sample': self._tensor_summary(sample),
                })
        return {
            'total_numel': total_numel,
            'trainable_numel': trainable_numel,
            'probe_count': len(probes),
            'probes': probes,
        }

    def _sample_parameter_indices(self, numel):
        sample_size = min(numel, self._alignment_full_param_sample_size)
        if sample_size == 0:
            return []
        if sample_size == numel:
            return list(range(numel))
        if sample_size == 1:
            return [0]
        indices = torch.arange(sample_size, dtype=torch.int64)
        indices = indices * (numel - 1) // (sample_size - 1)
        return indices.tolist()

    def _capture_full_parameter_samples(self):
        captured = {}
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and self._is_full_parameter_probe(name):
                    key = f'model{model_idx}.{name}'
                    captured[key] = {
                        'full_shape': list(parameter.shape),
                        'full_dtype': str(parameter.dtype),
                        'full_numel': parameter.numel(),
                        'sample': self._sample_parameter_tensor(parameter),
                    }
        return captured

    def _capture_trainable_tensors(self):
        if self.args.tuner_type == 'full':
            return self._capture_full_parameter_samples()
        captured = {}
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    captured[f'model{model_idx}.{name}'] = self._cpu_float_tensor(parameter)
        return captured

    def _full_parameter_update_summary(self, before):
        probe_parameters = []
        trainable_numel = 0
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                trainable_numel += parameter.numel()
                key = f'model{model_idx}.{name}'
                if key not in before:
                    continue
                after_sample = self._sample_parameter_tensor(parameter)
                before_sample = before[key]['sample']
                delta_sample = after_sample - before_sample
                grad = getattr(parameter, 'main_grad', None)
                if grad is None:
                    grad = parameter.grad
                grad_sample = self._sample_parameter_tensor(grad) if grad is not None else None
                probe_parameters.append({
                    'name': key,
                    'indices': self._sample_parameter_indices(parameter.numel()),
                    'gradient': self._tensor_summary(grad_sample, include_values=True),
                    'delta': self._tensor_summary(delta_sample, include_values=True),
                })
        return {
            'mode': 'full_parameter_sampled',
            'trainable_numel': trainable_numel,
            'parameters': probe_parameters,
        }

    def _trainable_update_summary(self, before):
        if self.args.tuner_type == 'full':
            return self._full_parameter_update_summary(before)
        param_values = []
        delta_values = []
        grad_values = []
        per_parameter = []
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                key = f'model{model_idx}.{name}'
                after = self._cpu_float_tensor(parameter)
                param_values.append(after.reshape(-1))
                if key in before:
                    delta_values.append((after - before[key]).reshape(-1))
                grad = getattr(parameter, 'main_grad', None)
                if grad is None:
                    grad = parameter.grad
                if grad is not None:
                    grad_values.append(self._cpu_float_tensor(grad).reshape(-1))
                if len(per_parameter) < 16:
                    per_parameter.append({
                        'name': key,
                        'parameter': self._tensor_summary(after),
                        'delta': self._tensor_summary(after - before[key]) if key in before else None,
                        'gradient': self._tensor_summary(grad) if grad is not None else None,
                    })
        concatenate = lambda values: torch.cat(values) if values else None
        return {
            'parameters': self._tensor_summary(concatenate(param_values)),
            'parameter_delta': self._tensor_summary(concatenate(delta_values)),
            'gradient_buffers': self._tensor_summary(concatenate(grad_values)),
            'first_parameters': per_parameter,
        }

    def train_step(self, train_data_iterator):
        step = int(self.state.iteration)
        debug_active = self._alignment_debug_active(step)
        before = self._capture_trainable_tensors() if debug_active else None
        result = super().train_step(train_data_iterator)
        if debug_active:
            _, grad_norm, update_successful = result
            learning_rate = next(
                (group['lr'] for group in self.optimizer.param_groups if group.get('params')), None)
            self._write_alignment_record({
                'record_type': 'optimizer_step',
                'step': step,
                'grad_norm': float(grad_norm) if grad_norm is not None else None,
                'learning_rate': float(learning_rate) if learning_rate is not None else None,
                'update_successful': bool(update_successful),
                'trainable_state': self._trainable_update_summary(before),
            })
        return result

    def train(self, train_dataset, val_dataset):
        if self.truncation_strategy == 'delete':
            self.resample_data_iterator = self._init_resample_data_iterator(train_dataset)
        super().train(train_dataset, val_dataset)

    def prepare_model(self):
        super().prepare_model()
        cache_mode = os.getenv('SWIFT_GKD_TEACHER_CACHE_MODE', '').lower()
        cache_reuse_step = os.getenv('SWIFT_GKD_TEACHER_CACHE_REUSE_STEP')
        if cache_mode == 'load' and cache_reuse_step is not None:
            if self.use_teacher_api:
                raise ValueError('teacher_model_server cannot be combined with teacher cache reuse.')
            self.teacher_models = []
            logger.info('Skipping local teacher model loading because a reusable teacher logits cache is enabled.')
            return
        if self.use_teacher_api or self._is_self_distillation:
            if self._is_self_distillation:
                logger.info('Self-distillation mode: using student model as teacher (no separate teacher loaded)')
            else:
                logger.info('Skipping local teacher model loading - using external API for teacher logprobs')
            return
        args = self.args
        vp_size = getattr(args, 'virtual_pipeline_model_parallel_size')
        assert vp_size is None or vp_size == 1, 'GKD currently does not support VPP.'
        self.teacher_hf_config = AutoConfig.from_pretrained(args.teacher_model_dir, trust_remote_code=True)
        self.teacher_models = get_mcore_model(args, self.teacher_hf_config)
        self.teacher_config = self.teacher_models[0].config
        if not args.use_cpu_initialization:
            # same as wrap_model in megatron_lm_utils.py
            for teacher_model in self.teacher_models:
                teacher_model.cuda(torch.cuda.current_device())
        for teacher_model in self.teacher_models:
            teacher_model.requires_grad_(False)
            teacher_model.eval()
        self.teacher_config.bridge.load_weights(self.teacher_models, args.teacher_model_dir)

        # Offload teacher models to CPU if enabled
        if self.offload_teacher_model:
            self._offload_teacher_models()
            logger.info('Teacher models offloaded to CPU to save GPU memory')

    def _offload_teacher_models(self):
        """Offload teacher models to CPU to save GPU memory."""
        if self.teacher_models and not self.use_teacher_api:
            offload_megatron_model_to_cpu(self.teacher_models)

    def _load_teacher_models_to_gpu(self):
        """Load teacher models back to GPU."""
        if self.teacher_models and not self.use_teacher_api:
            load_megatron_model_to_gpu(self.teacher_models, load_grad=False)

    @contextmanager
    def load_teacher_model_context(self):
        """Context manager to load teacher models for forward pass and optionally offload after.

        When offload_teacher_model is enabled:
        - Load teacher models to GPU before forward pass
        - Offload teacher models to CPU after forward pass

        This saves GPU memory during the training step.
        """
        if not self.offload_teacher_model:
            yield
            return

        self._load_teacher_models_to_gpu()
        try:
            yield
        finally:
            self._offload_teacher_models()

    @contextmanager
    def _template_context(self, template: Template, max_length: Optional[int] = None):
        """Context manager to temporarily modify max_length constraint from template."""
        original_max_length = template.max_length
        template.max_length = max_length
        try:
            yield
        finally:
            template.max_length = original_max_length

    def _build_opsd_teacher_data(self, inputs: List[Dict]) -> Optional[List[Dict]]:
        return build_opsd_teacher_data(inputs)

    def _encode_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Encode a batch of raw data into model inputs."""
        template = self.template
        args = self.args
        max_length = template.max_length + self.max_completion_length
        non_thinking_prefix_ids = get_non_thinking_prefix_ids(template)
        for data in batch:
            if 'response_token_ids' in data:
                data['messages'] = replace_assistant_response_with_ids(
                    data['messages'], data['response_token_ids'], non_thinking_prefix_ids=non_thinking_prefix_ids)

        with self._template_context(template, max_length=max_length):
            encoded_list = [template.encode(data, return_length=True) for data in batch]
            padding_to = get_padding_to(args)
            encoded_batch = to_device(template.data_collator(encoded_list, padding_to=padding_to), self.device)

        encoded_batch['num_samples'] = len(batch)
        return encoded_batch

    def _get_random_num(self) -> float:
        """Generate a deterministic random number consistent across all processes.

        Uses an isolated Random instance with seed based on args.seed + step counter

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self._step)
        rng = random.Random(seed)
        return rng.random()

    def _determine_data_source(self) -> DataSource:
        """Determine data source for current step based on GKD algorithm.

        GKD training mode selection logic:
        1. With probability lmbda: On-Policy (student generates)
        2. If seq_kd=True and not on-policy: Sequential KD (teacher generates)
        3. Otherwise: Off-Policy (use dataset responses)

        Returns:
            DataSource enum indicating which source to use.
        """
        random_num = self._get_random_num()

        if random_num < self.lmbda:
            # Mode 1: On-Policy learning, student model generates responses
            if self.use_vllm:
                return DataSource.STUDENT
            else:
                # If vLLM not enabled, fall back to dataset
                logger.warning_once('On-policy mode triggered but use_vllm=False. '
                                    'Falling back to dataset responses. Enable vLLM for on-policy generation.')
                return DataSource.DATASET
        elif self.seq_kd:
            # Mode 2: Sequential KD, teacher model generates responses
            # Note: Teacher generation is not implemented yet, use dataset
            logger.warning_once('seq_kd=True but teacher generation is not implemented in Megatron GKD yet. '
                                'Falling back to dataset responses.')
            return DataSource.DATASET
        else:
            # Mode 3: Off-Policy learning, use dataset responses
            return DataSource.DATASET

    def _init_resample_data_iterator(self, train_dataset):
        """Initialize an independent data iterator for resampling.

        Uses a different seed (args.seed + 1) to avoid overlapping with training samples.

        Args:
            train_dataset: The training dataset to create the resample iterator from.

        Returns:
            The resample data iterator (first element of the iterator tuple).
        """
        args = self.args
        resample_seed = getattr(args, 'seed', 42) + 1
        try:
            set_random_seed(
                resample_seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
            )
            resample_data_iterator = self._prepare_data_iterator(train_dataset, use_origin_cyclic=True)[0]
        finally:
            set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
            )
        return resample_data_iterator

    def resample_encode_failed_inputs(self, inputs: List[Dict], max_resample_rounds: int = 10) -> List[Dict]:
        """Attempt to encode each input. If encoding fails, resample until we have enough valid samples.

        Args:
            inputs: List of input data samples
            max_resample_rounds: Maximum number of resample rounds

        Returns:
            List of successfully encoded input samples with the same length as inputs
        """
        template = self.template
        required_count = len(inputs)
        valid_samples = []
        pending_samples = list(inputs)

        for _ in range(max_resample_rounds + 1):
            still_needed = required_count - len(valid_samples)
            if still_needed <= 0:
                break

            while len(pending_samples) < still_needed:
                pending_samples.extend(next(self.resample_data_iterator))

            while pending_samples and len(valid_samples) < required_count:
                data = pending_samples.pop(0)
                try:
                    template.encode(data)
                    valid_samples.append(data)
                except Exception as e:
                    logger.info(f'Encoding failed for one sample; will resample. {e}')

        if len(valid_samples) < required_count:
            raise RuntimeError(
                f'Failed to collect {required_count} valid samples after {max_resample_rounds} resample rounds. '
                f'Only collected {len(valid_samples)} valid samples. '
                'Consider increasing `max_length` or adjusting the `truncation_strategy`.')

        return valid_samples[:required_count]

    def _fetch_teacher_parsed_logprobs(self, raw_batch: List[Dict]):
        rollout_group = self._get_rollout_group()
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        contribution = list(raw_batch) if rollout_rank == 0 else []

        world_size = torch.distributed.get_world_size()
        all_contributions = [None] * world_size
        torch.distributed.all_gather_object(all_contributions, contribution)

        if self.is_main_process:
            flat_global = []
            for c in all_contributions:
                if c:
                    flat_global.extend(c)
            requests = [build_teacher_infer_request(d) for d in flat_global]
            request_config = RequestConfig(prompt_logprobs=self.gkd_logits_topk, max_tokens=1, temperature=0.0)
            responses = self.teacher_client.infer(requests, request_config=request_config, use_tqdm=False)
            parsed_global = [parse_prompt_logprobs(r, topk=self.gkd_logits_topk) for r in responses]
        else:
            parsed_global = None

        obj_list = [parsed_global]
        torch.distributed.broadcast_object_list(obj_list, src=world_size - 1)
        parsed_global = obj_list[0]

        # Slice for this DP partition. flat_global is concatenation of contributions in
        # ascending global_rank order; with TP/CP/PP-major rank layout, the canonical
        # rollout-rank-0 ranks form an ordered list aligned with data_parallel_rank.
        n = len(raw_batch)
        dp_rank = mpu.get_data_parallel_rank()
        return parsed_global[dp_rank * n:(dp_rank + 1) * n]

    def _assemble_teacher_outputs(self, encoded_batches: List[Dict]) -> None:
        """Build TeacherOutput from `_teacher_parsed` for each micro-batch.

        Simply uses encoded_batch's input_ids shape for the output tensor.
        For OPSD, stores rolled teacher labels for loss masking.
        """
        topk = self.gkd_logits_topk

        for encoded_batch in encoded_batches:
            parsed = encoded_batch.pop('_teacher_parsed')

            opsd_batch = encoded_batch.get('opsd_teacher_batch')
            source = opsd_batch if opsd_batch is not None else encoded_batch
            input_ids = source['input_ids']

            # Shape-based packed detection: [1, T] with multiple seqs vs [B, S]
            batch_size, seq_len = input_ids.shape
            server_seq_lens = None
            if self.template.padding_free:
                server_seq_lens = [0]
                for lps, ixs in parsed:
                    server_seq_lens.append(server_seq_lens[-1] + len(lps) + 1)
                trainer_seq_lens = encoded_batch.get('cu_seq_lens_q')
                if trainer_seq_lens is None:
                    position_ids = encoded_batch.get('text_position_ids')
                    if position_ids is None:
                        position_ids = encoded_batch.get('position_ids')
                    if position_ids is not None:
                        trainer_seq_lens = get_cu_seqlens_from_position_ids(position_ids)
                if trainer_seq_lens is not None and server_seq_lens[-1] != int(trainer_seq_lens[-1]):
                    logger.warning(
                        'The number of tokens returned by the teacher server differs from that of the trainer. '
                        'This may be caused by non-aligned processing')
            topk_logprobs, topk_indices = assemble_teacher_topk_logprobs(
                parsed,
                batch_size=batch_size,
                seq_len=seq_len,
                cu_seqlens=server_seq_lens,
                topk=topk,
                device=self.device)

            teacher_out = TeacherOutput(topk_logprobs=topk_logprobs, topk_indices=topk_indices)
            if opsd_batch is not None:
                teacher_out.opsd_teacher_labels = torch.roll(opsd_batch['labels'], shifts=-1, dims=-1)
            encoded_batch['teacher_output'] = teacher_out

    def _compute_teacher_logits(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        if self.use_teacher_api:
            self._assemble_teacher_outputs(encoded_batches)
            return
        self._compute_teacher_logits_local(encoded_batches, vp_stage)

    def _compute_teacher_logits_local(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        topk = self.gkd_logits_topk

        cache_only = self._teacher_cache_mode == 'load' and self._teacher_cache_reuse_step is not None
        if cache_only:
            teacher_model = None
            outer_context = ContextManagers([])
        elif self._is_self_distillation:
            teacher_model = self.unwrapped_models[0]
            adapter_contexts = []
            if self._teacher_use_disable_adapter:
                adapter_contexts = [m.disable_adapter() for m in self.peft_models]
            outer_context = ContextManagers(adapter_contexts)
        else:
            teacher_model = self.teacher_models[vp_stage or 0]
            outer_context = self.load_teacher_model_context()

        with torch.no_grad(), outer_context:
            for teacher_micro_idx, encoded_batch in enumerate(encoded_batches):
                opsd_batch = encoded_batch.get('opsd_teacher_batch')
                source = opsd_batch if opsd_batch is not None else encoded_batch
                teacher_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in source.items() if k not in ('data_source', 'opsd_teacher_batch', 'teacher_output')
                }
                teacher_data = self._prepare_batch(teacher_batch)
                teacher_data.pop('loss_scale', None)
                opsd_teacher_labels = teacher_data.pop('labels', None)
                if opsd_batch is None:
                    opsd_teacher_labels = None
                step = int(self.state.iteration)
                cache_active = self._teacher_cache_mode and (
                    self._teacher_cache_reuse_step is not None
                    or step < max(self._alignment_debug_steps, 1))
                cache_path = None
                if cache_active:
                    if mpu.get_tensor_model_parallel_world_size() != 1:
                        raise ValueError('Teacher logits cache isolation currently requires tensor parallel size 1.')
                    cache_step = (
                        self._teacher_cache_reuse_step
                        if self._teacher_cache_reuse_step is not None else step)
                    cache_path = os.path.join(
                        self._teacher_cache_dir, f'teacher_step_{cache_step:06d}_micro_{teacher_micro_idx:03d}.pt')
                if cache_active and self._teacher_cache_mode == 'load':
                    if not os.path.isfile(cache_path):
                        raise FileNotFoundError(f'Teacher logits cache not found: {cache_path}')
                    payload = torch.load(cache_path, map_location='cpu', weights_only=True)
                    target_device = next(v.device for v in teacher_data.values() if isinstance(v, torch.Tensor))
                    teacher_logits = payload['teacher_logits'].to(target_device)
                    input_ids = teacher_data.get('input_ids')
                    if input_ids is not None and tuple(teacher_logits.shape[:2]) != tuple(input_ids.shape):
                        raise ValueError(
                            f'Cached teacher logits shape {tuple(teacher_logits.shape)} does not match '
                            f'input_ids shape {tuple(input_ids.shape)} at step={step}, micro={teacher_micro_idx}.')
                else:
                    teacher_logits = forward_step_helper(teacher_model, teacher_data)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()
                if cache_active and self._teacher_cache_mode == 'save' and teacher_logits is not None:
                    torch.save({'teacher_logits': teacher_logits.cpu()}, cache_path)

                if topk is not None and teacher_logits is not None:
                    topk_logits, topk_indices = vocab_parallel_topk(teacher_logits, k=topk)
                    teacher_out = TeacherOutput(topk_logprobs=topk_logits, topk_indices=topk_indices)
                else:
                    teacher_out = TeacherOutput(full_logits=teacher_logits)

                teacher_out.opsd_teacher_labels = opsd_teacher_labels
                encoded_batch['teacher_output'] = teacher_out

    def _replace_data_iterator(self, data_iterator):
        num_microbatches = self.args.num_microbatches
        steps_per_generation = self.steps_per_generation

        if self._step % steps_per_generation == 0:
            data_source = self._determine_data_source()

            total_microbatches = num_microbatches * steps_per_generation
            global_batch = []
            for _ in range(total_microbatches):
                raw_batch = next(data_iterator)
                if self.truncation_strategy == 'delete' and self.resample_data_iterator is not None:
                    raw_batch = self.resample_encode_failed_inputs(raw_batch)
                global_batch.extend(raw_batch)

            if data_source == DataSource.STUDENT:
                local_batch = self._get_local_rollout_batch(global_batch)
                local_batch = self._generate_completions(local_batch)
                global_batch = self._gather_rollout_results(local_batch)
            elif data_source == DataSource.TEACHER:
                logger.warning_once(
                    'Teacher mode triggered but teacher generation is not implemented in Megatron GKD yet. '
                    'Falling back to dataset responses.')

            teacher_global_batch = self._build_opsd_teacher_data(global_batch)

            # Fetch teacher prompt_logprobs once per global batch when using teacher API.
            # OPSD: use teacher-prompt batch; otherwise: use the regular global batch.
            local_parsed = None
            if self.use_teacher_api:
                teacher_raw = teacher_global_batch if teacher_global_batch is not None else global_batch
                local_parsed = self._fetch_teacher_parsed_logprobs(teacher_raw)

            micro_batch_size = len(global_batch) // total_microbatches
            assert micro_batch_size == self.args.micro_batch_size
            all_encoded_batches = []
            for i in range(total_microbatches):
                start_idx = i * micro_batch_size
                end_idx = start_idx + micro_batch_size
                raw_batch = global_batch[start_idx:end_idx]
                encoded_batch = self._encode_batch(raw_batch)
                encoded_batch['data_source'] = data_source
                if teacher_global_batch is not None:
                    teacher_slice = teacher_global_batch[start_idx:end_idx]
                    encoded_batch['opsd_teacher_batch'] = self._encode_batch(teacher_slice)
                if local_parsed is not None:
                    encoded_batch['_teacher_parsed'] = local_parsed[start_idx:end_idx]
                all_encoded_batches.append(encoded_batch)
            self._compute_teacher_logits(all_encoded_batches)

            self._buffered_inputs = [
                all_encoded_batches[i * num_microbatches:(i + 1) * num_microbatches]
                for i in range(steps_per_generation)
            ]

        step_idx = self._step % steps_per_generation
        encoded_batches = self._buffered_inputs[step_idx]
        self._step += 1

        return RerunDataIterator(iter(encoded_batches))

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  teacher_output: TeacherOutput,
                  data_source: DataSource = DataSource.DATASET):
        """Compute GKD loss (JSD + optional SFT loss)."""
        student_logits = output_tensor

        jsd_total, jsd_num_valid = gkd_loss(
            student_logits,
            teacher_output,
            labels,
            self.beta,
            self.temperature,
            gather_fn=tp_gather_topk,
            log_softmax_fn=vocab_parallel_log_softmax,
            kl_div_fn=vocab_parallel_kl_div)
        jsd_loss_val = cp_reduce(jsd_total, jsd_num_valid, cp_size=self.args.context_parallel_size)

        debug_context = getattr(self, '_alignment_loss_context', None)
        if debug_context is not None:
            debug_context['jsd_total'] = float(jsd_total.detach().float().cpu())
            debug_context['jsd_num_valid'] = int(jsd_num_valid.detach().cpu())
            debug_context['jsd_loss'] = float(jsd_loss_val.detach().float().cpu())

        loss = jsd_loss_val

        # Add SFT loss if enabled (skip for student-generated responses)
        sft_loss = None
        if self.sft_alpha > 0 and data_source != DataSource.STUDENT:
            args = self.args
            logits_sbv = student_logits.transpose(0, 1).contiguous()
            model = self.unwrapped_models[0]
            if hasattr(model, 'language_model'):
                model = model.language_model
            per_token_loss = model.compute_language_model_loss(labels, logits_sbv)
            loss_mask = labels != -100
            sft_loss_sum = (per_token_loss * loss_mask).sum()
            sft_loss_count = loss_mask.sum().float()

            # All-reduce across CP group for correct averaging
            if args.context_parallel_size > 1:
                sft_stats = torch.stack([sft_loss_sum, sft_loss_count])
                torch.distributed.all_reduce(
                    sft_stats, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
                sft_loss_sum, sft_loss_count = sft_stats[0], sft_stats[1]

            sft_loss = sft_loss_sum / sft_loss_count

            loss = loss + self.sft_alpha * sft_loss

        metric = {'loss': loss.detach().clone()}
        if sft_loss is not None:
            metric['jsd_loss'] = jsd_loss_val.detach().clone()
            metric['sft_loss'] = sft_loss.detach().clone()
        metric = self._all_reduce_metric(metric)

        loss = loss / mpu.get_context_parallel_world_size()

        return loss, metric

    def forward_step(self, data_iterator, model):
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage

        data = next(data_iterator)
        data_source = data.pop('data_source', DataSource.DATASET)
        teacher_output = data.pop('teacher_output', TeacherOutput())
        data.pop('opsd_teacher_batch', None)
        data = self._prepare_batch(data, vp_stage)

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)

        step = int(self.state.iteration)
        micro_idx = self._alignment_micro_counts.get(step, 0)
        self._alignment_micro_counts[step] = micro_idx + 1
        debug_active = self._alignment_debug_active(step)
        if debug_active:
            self._alignment_loss_context = {}

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        if debug_active and (self._operator_debug_enabled or self._linear_proj_isolation_mode):
            self._operator_debug_context = {
                'step': step,
                'micro_batch': micro_idx,
                'call_counts': {},
            }
        try:
            student_output = model(**data)
        finally:
            self._operator_debug_context = None

        if debug_active:
            teacher_logits = teacher_output.full_logits
            teacher_labels = teacher_output.opsd_teacher_labels if teacher_output.opsd_teacher_labels is not None else labels
            record = {
                'record_type': 'forward',
                'step': step,
                'micro_batch': micro_idx,
                'input_ids': self._tensor_identity(data.get('input_ids')),
                'position_ids': self._tensor_identity(data.get('position_ids')),
                'labels': self._tensor_identity(labels),
                'num_valid': int((labels != -100).sum().item()) if labels is not None else None,
                'student_logits': self._logits_summary(student_output, labels),
                'teacher_logits': self._logits_summary(teacher_logits, teacher_labels),
            }
            if teacher_logits is None:
                record['teacher_topk_logprobs'] = self._tensor_summary(teacher_output.topk_logprobs)
                record['teacher_topk_indices'] = self._tensor_identity(teacher_output.topk_indices)

            def write_loss_record():
                record['loss'] = self._alignment_loss_context
                self._write_alignment_record(record)
                self._alignment_loss_context = None

            loss_callback = partial(
                self.loss_func,
                labels=labels,
                teacher_output=teacher_output,
                data_source=data_source,
            )

            def debug_loss_callback(output_tensor):
                result = loss_callback(output_tensor)
                write_loss_record()
                return result

            return student_output, debug_loss_callback

        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_output=teacher_output,
            data_source=data_source,
        )
