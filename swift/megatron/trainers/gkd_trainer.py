# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import inspect
import json
import os
import random
import re
import torch
import torch.nn.functional as F
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial
from mcore_bridge import set_random_seed
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from transformers import AutoConfig
from transformers.utils import ContextManagers
from typing import Any, Dict, List, Optional

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
from .gkd_attention_forward_debug import GKDAttentionForwardTrace
from .gkd_microbatch_backward_debug import GKDMicrobatchBackwardTrace
from .gkd_utils import cp_reduce, tp_gather_topk, vocab_parallel_topk
from .gkd_mlp_merge_debug import GKDMLPMergeIsolation
from .gkd_self_attention_forward_debug import GKDSelfAttentionForwardIsolation
from .gkd_runtime_audit import update_runtime_audit_grad_dtypes, write_runtime_audit
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from .vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax

logger = get_logger()


class MegatronGKDTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    def __init__(self, args: MegatronArguments, template, **kwargs):
        self._strict_fp32 = os.getenv('SWIFT_GKD_STRICT_FP32', '0') == '1'
        if self._strict_fp32:
            self._validate_strict_fp32_args(args)
        # Controlled runtime audit: disable activation recomputation before the
        # base trainer constructs the model/configuration.
        if os.getenv('SWIFT_GKD_AUDIT_DISABLE_CORE_ATTN_RECOMPUTE', '0') == '1':
            # Megatron TransformerConfig expects None for disabled recompute;
            # the CLI spelling "none" is normalized earlier during argument parsing.
            args.recompute_granularity = None
            args.recompute_modules = []
            logger.info('GKD runtime audit: core attention recompute disabled')
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
        if self._strict_fp32:
            self._validate_strict_fp32_config()

        self._alignment_debug_start_step = int(os.getenv('SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP', '0'))
        self._alignment_debug_steps = int(os.getenv('SWIFT_GKD_ALIGNMENT_DEBUG_STEPS', '0'))
        if self._alignment_debug_start_step < 0:
            raise ValueError('SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP must be non-negative.')
        if self._alignment_debug_steps < self._alignment_debug_start_step:
            raise ValueError(
                'SWIFT_GKD_ALIGNMENT_DEBUG_STEPS must be greater than or equal to '
                'SWIFT_GKD_ALIGNMENT_DEBUG_START_STEP.')
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
        self._dtype_audit_enabled = os.getenv('SWIFT_GKD_DTYPE_AUDIT', '0') == '1'
        self._dtype_audit_model_done = False
        self._dtype_audit_grad_done = False
        self._dtype_audit_optimizer_done = False
        self._grad_clip_debug_dir = os.getenv('SWIFT_GKD_GRAD_CLIP_DEBUG_DIR')
        self._grad_clip_debug_tag = os.getenv('SWIFT_GKD_GRAD_CLIP_DEBUG_TAG', '').strip()
        self._grad_clip_debug_start_step = int(os.getenv('SWIFT_GKD_GRAD_CLIP_DEBUG_START_STEP', '0'))
        self._grad_clip_debug_steps = int(os.getenv('SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS', '0'))
        if self._grad_clip_debug_start_step < 0:
            raise ValueError('SWIFT_GKD_GRAD_CLIP_DEBUG_START_STEP must be non-negative.')
        if self._grad_clip_debug_steps < self._grad_clip_debug_start_step:
            raise ValueError(
                'SWIFT_GKD_GRAD_CLIP_DEBUG_STEPS must be greater than or equal to '
                'SWIFT_GKD_GRAD_CLIP_DEBUG_START_STEP.')
        if self._grad_clip_debug_steps > 0 and not self._grad_clip_debug_dir:
            self._grad_clip_debug_dir = os.path.join(args.output_dir, 'gkd_grad_clip_debug')
        self._preclip_grad_dir = os.getenv('SWIFT_GKD_PRECLIP_GRAD_DIR')
        self._preclip_grad_tag = os.getenv('SWIFT_GKD_PRECLIP_GRAD_TAG', '').strip()
        preclip_steps = os.getenv('SWIFT_GKD_PRECLIP_GRAD_STEPS', '').strip()
        try:
            parsed_preclip_steps = [int(value.strip()) for value in preclip_steps.split(',') if value.strip()]
        except ValueError as error:
            raise ValueError('SWIFT_GKD_PRECLIP_GRAD_STEPS must contain comma-separated integers.') from error
        if any(step < 0 for step in parsed_preclip_steps):
            raise ValueError('SWIFT_GKD_PRECLIP_GRAD_STEPS must contain only non-negative integers.')
        if len(parsed_preclip_steps) != len(set(parsed_preclip_steps)):
            raise ValueError('SWIFT_GKD_PRECLIP_GRAD_STEPS must not contain duplicate steps.')
        self._preclip_grad_steps = set(parsed_preclip_steps)
        self._preclip_grad_chunk_numel = int(
            os.getenv('SWIFT_GKD_PRECLIP_GRAD_CHUNK_NUMEL', str(4 * 1024 * 1024)))
        if self._preclip_grad_chunk_numel <= 0:
            raise ValueError('SWIFT_GKD_PRECLIP_GRAD_CHUNK_NUMEL must be greater than 0.')
        if self._preclip_grad_steps:
            if not self._preclip_grad_dir:
                raise ValueError('SWIFT_GKD_PRECLIP_GRAD_DIR is required when gradient capture is enabled.')
            if not self._preclip_grad_tag:
                raise ValueError('SWIFT_GKD_PRECLIP_GRAD_TAG is required when gradient capture is enabled.')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', self._preclip_grad_tag):
                raise ValueError('SWIFT_GKD_PRECLIP_GRAD_TAG may contain only letters, digits, _ and -.')
            logger.info(
                f'GKD full pre-clip gradient capture enabled: tag={self._preclip_grad_tag}, '
                f'steps={sorted(self._preclip_grad_steps)}, dir={self._preclip_grad_dir}')
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
        self._backward_debug_enabled = os.getenv(
            'SWIFT_GKD_BACKWARD_DEBUG', '0').lower() in {'1', 'true', 'yes'}
        backward_patterns = os.getenv(
            'SWIFT_GKD_BACKWARD_DEBUG_PATTERNS',
            'decoder.layers.0.self_attention.linear_qkv,decoder.layers.2.mlp.linear_fc2')
        self._backward_debug_patterns = [
            pattern.strip() for pattern in backward_patterns.split(',') if pattern.strip()
        ]
        self._backward_debug_context = None
        self._backward_debug_handles = []
        self._backward_debug_written = set()
        self._jsd_isolation_mode = os.getenv('SWIFT_GKD_JSD_ISOLATION_MODE', '').lower()
        self._jsd_isolation_dir = os.getenv('SWIFT_GKD_JSD_ISOLATION_DIR')
        self._jsd_isolation_step = int(os.getenv('SWIFT_GKD_JSD_ISOLATION_STEP', '13'))
        self._jsd_isolation_micro_batch = int(os.getenv('SWIFT_GKD_JSD_ISOLATION_MICRO_BATCH', '1'))
        self._jsd_isolation_done = False
        if self._jsd_isolation_mode not in {'', 'capture'}:
            raise ValueError('SWIFT_GKD_JSD_ISOLATION_MODE must be empty or "capture".')
        if self._jsd_isolation_mode and not self._jsd_isolation_dir:
            raise ValueError('SWIFT_GKD_JSD_ISOLATION_DIR is required for JSD isolation capture.')
        if self._jsd_isolation_step < 0 or self._jsd_isolation_micro_batch < 0:
            raise ValueError('JSD isolation step and micro-batch must be non-negative.')
        self._dlogits_isolation_mode = os.getenv('SWIFT_GKD_DLOGITS_ISOLATION_MODE', '').lower()
        self._dlogits_isolation_dir = os.getenv('SWIFT_GKD_DLOGITS_ISOLATION_DIR')
        self._dlogits_isolation_step = int(os.getenv('SWIFT_GKD_DLOGITS_ISOLATION_STEP', '0'))
        self._dlogits_isolation_micro_batch = int(os.getenv('SWIFT_GKD_DLOGITS_ISOLATION_MICRO_BATCH', '0'))
        self._dlogits_isolation_tag = os.getenv('SWIFT_GKD_DLOGITS_ISOLATION_TAG', '').lower()
        dlogits_patterns = os.getenv(
            'SWIFT_GKD_DLOGITS_BACKWARD_PATTERNS',
            'output_layer,decoder.layers.27,decoder.layers.2,decoder.layers.0')
        self._dlogits_backward_patterns = [
            pattern.strip() for pattern in dlogits_patterns.split(',') if pattern.strip()
        ]
        parameter_patterns = os.getenv(
            'SWIFT_GKD_DLOGITS_PARAMETER_PATTERNS',
            'output_layer.weight,decoder.layers.27.mlp.linear_fc2.weight,'
            'decoder.layers.2.mlp.linear_fc2.weight,'
            'decoder.layers.0.self_attention.linear_qkv.weight')
        self._dlogits_parameter_patterns = [
            pattern.strip() for pattern in parameter_patterns.split(',') if pattern.strip()
        ]
        self._dlogits_module_gradients = {}
        self._dlogits_provenance = None
        self._dlogits_hook_done = False
        if self._dlogits_isolation_mode not in {'', 'capture', 'replay'}:
            raise ValueError('SWIFT_GKD_DLOGITS_ISOLATION_MODE must be empty, "capture", or "replay".')
        if self._dlogits_isolation_mode and not self._dlogits_isolation_dir:
            raise ValueError('SWIFT_GKD_DLOGITS_ISOLATION_DIR is required for dLogits isolation.')
        if self._dlogits_isolation_step < 0 or self._dlogits_isolation_micro_batch < 0:
            raise ValueError('dLogits isolation step and micro-batch must be non-negative.')
        if self._dlogits_isolation_mode:
            self._backward_debug_patterns = list(dict.fromkeys(
                self._backward_debug_patterns + self._dlogits_backward_patterns))
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
        self._flash_isolation_mode = os.getenv('SWIFT_GKD_FLASH_ISOLATION_MODE', '').lower()
        self._flash_isolation_dir = os.getenv('SWIFT_GKD_FLASH_ISOLATION_DIR')
        self._flash_isolation_prefix = os.getenv(
            'SWIFT_GKD_FLASH_ISOLATION_TARGET_PREFIX',
            'decoder.layers.0.self_attention.core_attention')
        self._flash_isolation_tag = os.getenv('SWIFT_GKD_FLASH_ISOLATION_TAG', '').lower()
        self._flash_isolation_step = int(os.getenv('SWIFT_GKD_FLASH_ISOLATION_STEP', '0'))
        self._flash_isolation_micro_batch = int(os.getenv('SWIFT_GKD_FLASH_ISOLATION_MICRO_BATCH', '0'))
        self._flash_backward_isolation_enabled = os.getenv(
            'SWIFT_GKD_FLASH_BACKWARD_ISOLATION', '0').lower() in {'1', 'true', 'yes'}
        self._flash_backward_isolation_done = False
        self._flash_isolation_done = False
        self._flash_isolation_handles = []
        if self._flash_isolation_mode not in {'', 'capture', 'replay'}:
            raise ValueError('SWIFT_GKD_FLASH_ISOLATION_MODE must be empty, "capture", or "replay".')
        if self._flash_isolation_mode and not self._flash_isolation_dir:
            raise ValueError('SWIFT_GKD_FLASH_ISOLATION_DIR is required for Flash Attention isolation.')
        if self._flash_backward_isolation_enabled and not self._flash_isolation_mode:
            raise ValueError(
                'SWIFT_GKD_FLASH_BACKWARD_ISOLATION requires Flash Attention capture or replay mode.')
        if self._flash_isolation_step < 0 or self._flash_isolation_micro_batch < 0:
            raise ValueError('Flash Attention isolation step and micro-batch must be non-negative.')
        self._swiglu_isolation_mode = os.getenv('SWIFT_GKD_SWIGLU_ISOLATION_MODE', '').lower()
        self._swiglu_isolation_dir = os.getenv('SWIFT_GKD_SWIGLU_ISOLATION_DIR')
        self._swiglu_isolation_target = os.getenv(
            'SWIFT_GKD_SWIGLU_ISOLATION_TARGET',
            'decoder.layers.2.mlp.linear_fc1')
        default_swiglu_dout_target = self._swiglu_isolation_target.replace('linear_fc1', 'linear_fc2')
        self._swiglu_isolation_dout_target = os.getenv(
            'SWIFT_GKD_SWIGLU_ISOLATION_DOUT_TARGET', default_swiglu_dout_target)
        self._swiglu_isolation_step = int(os.getenv('SWIFT_GKD_SWIGLU_ISOLATION_STEP', '0'))
        self._swiglu_isolation_micro_batch = int(os.getenv('SWIFT_GKD_SWIGLU_ISOLATION_MICRO_BATCH', '0'))
        self._swiglu_isolation_input_captured = False
        self._swiglu_isolation_payload = None
        self._swiglu_isolation_done = False
        self._swiglu_isolation_handles = []
        if self._swiglu_isolation_mode not in {'', 'capture'}:
            raise ValueError('SWIFT_GKD_SWIGLU_ISOLATION_MODE must be empty or "capture".')
        if self._swiglu_isolation_mode and not self._swiglu_isolation_dir:
            raise ValueError('SWIFT_GKD_SWIGLU_ISOLATION_DIR is required for SwiGLU isolation.')
        if self._swiglu_isolation_step < 0 or self._swiglu_isolation_micro_batch < 0:
            raise ValueError('SwiGLU isolation step and micro-batch must be non-negative.')
        self._fc1_isolation_mode = os.getenv('SWIFT_GKD_FC1_ISOLATION_MODE', '').lower()
        self._fc1_isolation_dir = os.getenv('SWIFT_GKD_FC1_ISOLATION_DIR')
        self._fc1_isolation_target = os.getenv(
            'SWIFT_GKD_FC1_ISOLATION_TARGET',
            'decoder.layers.27.mlp.linear_fc1')
        self._fc1_isolation_step = int(os.getenv('SWIFT_GKD_FC1_ISOLATION_STEP', '0'))
        self._fc1_isolation_micro_batch = int(os.getenv('SWIFT_GKD_FC1_ISOLATION_MICRO_BATCH', '0'))
        self._fc1_isolation_tag = os.getenv('SWIFT_GKD_FC1_ISOLATION_TAG', '').lower()
        self._fc1_isolation_handles = []
        self._fc1_isolation_module = None
        self._fc1_isolation_common = None
        self._fc1_isolation_result = None
        self._fc1_isolation_expected_output_gradients = set()
        self._fc1_isolation_seen_output_gradients = set()
        self._fc1_isolation_forward_done = False
        if self._fc1_isolation_mode not in {'', 'capture', 'replay'}:
            raise ValueError('SWIFT_GKD_FC1_ISOLATION_MODE must be empty, "capture", or "replay".')
        if self._fc1_isolation_mode and not self._fc1_isolation_dir:
            raise ValueError('SWIFT_GKD_FC1_ISOLATION_DIR is required for FC1 isolation.')
        if self._fc1_isolation_step < 0 or self._fc1_isolation_micro_batch < 0:
            raise ValueError('FC1 isolation step and micro-batch must be non-negative.')
        self._mlp_merge_isolation = GKDMLPMergeIsolation(self)
        self._self_attention_forward_isolation = GKDSelfAttentionForwardIsolation(self)
        self._attention_forward_trace = GKDAttentionForwardTrace(self)
        self._microbatch_backward_trace = GKDMicrobatchBackwardTrace(self)
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
            logger.info(
                f'GKD alignment debug step range: '
                f'[{self._alignment_debug_start_step}, {self._alignment_debug_steps})')
        if self._grad_clip_debug_steps > 0 and self._is_debug_rank():
            os.makedirs(self._grad_clip_debug_dir, exist_ok=True)
            logger.info(f'GKD gradient clipping debug output: {self._grad_clip_debug_path}')
            logger.info(
                f'GKD gradient clipping debug step range: '
                f'[{self._grad_clip_debug_start_step}, {self._grad_clip_debug_steps})')
        if self._teacher_cache_mode and self._is_debug_rank():
            os.makedirs(self._teacher_cache_dir, exist_ok=True)
            logger.info(f'GKD teacher cache mode={self._teacher_cache_mode}, dir={self._teacher_cache_dir}')
            if self._teacher_cache_reuse_step is not None:
                logger.warning(
                    f'Reusing teacher logits from cache step {self._teacher_cache_reuse_step} for every step. '
                    'The input IDs and micro-batch order must repeat exactly.')
        if self._flash_isolation_mode and self._is_debug_rank():
            os.makedirs(self._flash_isolation_dir, exist_ok=True)
            logger.info(f'GKD Flash Attention isolation output directory ready: {self._flash_isolation_dir}')
        if self._dlogits_isolation_mode and self._is_debug_rank():
            os.makedirs(self._dlogits_isolation_dir, exist_ok=True)
            logger.info(
                f'GKD common dLogits isolation mode={self._dlogits_isolation_mode}, '
                f'dir={self._dlogits_isolation_dir}')
        if self._fc1_isolation_mode and self._is_debug_rank():
            os.makedirs(self._fc1_isolation_dir, exist_ok=True)
            logger.info(
                f'GKD FC1 isolation mode={self._fc1_isolation_mode}, '
                f'target={self._fc1_isolation_target}, dir={self._fc1_isolation_dir}')

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
        # Hooks must be installed before training even when the debug window starts after step 0.
        if self._alignment_debug_steps > self._alignment_debug_start_step and self._operator_debug_enabled:
            self._register_operator_debug_hooks()
        if (self._alignment_debug_steps > self._alignment_debug_start_step
                and (self._backward_debug_enabled or self._dlogits_isolation_mode)):
            self._register_backward_debug_hooks()
        if self._alignment_debug_active(0) and self._linear_proj_isolation_mode:
            self._register_linear_proj_isolation_hooks()
        if self._flash_isolation_mode:
            self._register_flash_isolation_hooks()
        if self._swiglu_isolation_mode:
            self._register_swiglu_isolation_hooks()
        if self._fc1_isolation_mode:
            self._register_fc1_isolation_hooks()
        if self._mlp_merge_isolation.enabled:
            self._mlp_merge_isolation.register_hooks()
        if self._self_attention_forward_isolation.enabled:
            self._self_attention_forward_isolation.register_hooks()
        if self._attention_forward_trace.enabled:
            self._attention_forward_trace.register_hooks()
        if self._microbatch_backward_trace.enabled:
            self._microbatch_backward_trace.register_hooks()

    @staticmethod
    def _validate_strict_fp32_args(args):
        checks = {
            'torch_dtype': (args.torch_dtype, torch.float32),
            'fp16': (args.fp16, False),
            'bf16': (args.bf16, False),
            'attention_softmax_in_fp32': (args.attention_softmax_in_fp32, True),
            'use_precision_aware_optimizer': (args.use_precision_aware_optimizer, False),
            'main_grads_dtype': (args.main_grads_dtype, torch.float32),
            'main_params_dtype': (args.main_params_dtype, torch.float32),
            'exp_avg_dtype': (args.exp_avg_dtype, torch.float32),
            'exp_avg_sq_dtype': (args.exp_avg_sq_dtype, torch.float32),
            'accumulate_allreduce_grads_in_fp32': (args.accumulate_allreduce_grads_in_fp32, True),
            'SWIFT_GKD_JSD_FP32': (os.getenv('SWIFT_GKD_JSD_FP32'), '1'),
        }
        mismatches = [
            f'{name}={actual!r} (expected {expected!r})'
            for name, (actual, expected) in checks.items() if actual != expected
        ]
        if getattr(args, 'fp8_format', None) is not None:
            mismatches.append(f'fp8_format={args.fp8_format!r} (expected None)')
        if mismatches:
            raise ValueError('SWIFT_GKD_STRICT_FP32 configuration mismatch: ' + '; '.join(mismatches))

    def _validate_strict_fp32_config(self):
        config = self.config
        checks = {
            'config.params_dtype': (getattr(config, 'params_dtype', None), torch.float32),
            'config.fp32_residual_connection': (
                getattr(config, 'fp32_residual_connection', False), True),
        }
        mismatches = [
            f'{name}={actual!r} (expected {expected!r})'
            for name, (actual, expected) in checks.items() if actual != expected
        ]
        if mismatches:
            raise ValueError('SWIFT_GKD_STRICT_FP32 model config mismatch: ' + '; '.join(mismatches))
        logger.info('GKD strict FP32 configuration validated.')

    @staticmethod
    def _strict_fp32_tensor_error(name, tensor):
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dtype != torch.float32:
            return f'{name}={tensor.dtype}'
        return None

    def _validate_strict_fp32_models(self, label, models):
        if not self._strict_fp32 or not models:
            return
        errors = []
        for model_idx, model in enumerate(models):
            for kind, named_tensors in (
                    ('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
                for name, tensor in named_tensors:
                    error = self._strict_fp32_tensor_error(
                        f'{label}[{model_idx}].{kind}.{name}', tensor)
                    if error is not None:
                        errors.append(error)
                        if len(errors) >= 8:
                            break
                if len(errors) >= 8:
                    break
            if len(errors) >= 8:
                break
        if errors:
            raise RuntimeError('SWIFT_GKD_STRICT_FP32 model dtype mismatch: ' + '; '.join(errors))

    def _validate_strict_fp32_tensors(self, **tensors):
        if not self._strict_fp32:
            return
        errors = [self._strict_fp32_tensor_error(name, tensor) for name, tensor in tensors.items()]
        errors = [error for error in errors if error is not None]
        if errors:
            raise RuntimeError('SWIFT_GKD_STRICT_FP32 tensor dtype mismatch: ' + '; '.join(errors))

    def _validate_strict_fp32_gradients(self):
        if not self._strict_fp32:
            return
        errors = []
        for model_idx, model in enumerate(self.unwrapped_models or []):
            for name, parameter in model.named_parameters():
                for grad_name, gradient in (
                        ('grad', parameter.grad), ('main_grad', getattr(parameter, 'main_grad', None))):
                    error = self._strict_fp32_tensor_error(
                        f'student[{model_idx}].{name}.{grad_name}', gradient)
                    if error is not None:
                        errors.append(error)
                        if len(errors) >= 8:
                            break
                if len(errors) >= 8:
                    break
            if len(errors) >= 8:
                break
        if errors:
            raise RuntimeError('SWIFT_GKD_STRICT_FP32 gradient dtype mismatch: ' + '; '.join(errors))

    def _validate_strict_fp32_optimizer(self):
        if not self._strict_fp32:
            return
        errors = []

        def inspect_value(name, value):
            error = self._strict_fp32_tensor_error(name, value)
            if error is not None:
                errors.append(error)
            elif isinstance(value, Mapping):
                for key, child in value.items():
                    inspect_value(f'{name}.{key}', child)
                    if len(errors) >= 8:
                        return
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    inspect_value(f'{name}[{index}]', child)
                    if len(errors) >= 8:
                        return

        parameter_idx = 0
        for group in self.optimizer.param_groups:
            for parameter in group.get('params', []):
                inspect_value(f'optimizer.parameter[{parameter_idx}]', parameter)
                inspect_value(f'optimizer.main_param[{parameter_idx}]', getattr(parameter, 'main_param', None))
                try:
                    state = self.optimizer.state[parameter]
                except (KeyError, TypeError, AttributeError):
                    state = None
                inspect_value(f'optimizer.state[{parameter_idx}]', state)
                parameter_idx += 1
                if len(errors) >= 8:
                    break
            if len(errors) >= 8:
                break
        if errors:
            raise RuntimeError('SWIFT_GKD_STRICT_FP32 optimizer dtype mismatch: ' + '; '.join(errors))

    @property
    def _alignment_debug_path(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return os.path.join(self._alignment_debug_dir, f'rank{rank}_alignment.jsonl')

    @property
    def _grad_clip_debug_path(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return os.path.join(self._grad_clip_debug_dir, f'rank{rank}_grad_clip.jsonl')

    @staticmethod
    def _is_debug_rank():
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    @staticmethod
    def _is_preclip_grad_capture_rank():
        # Gradients are replicated across the data-parallel group. Keep DP rank 0,
        # while retaining every tensor/pipeline partition when model parallelism is used.
        return not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0

    def _capture_preclip_gradients(self, step):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        step_dir = os.path.join(
            self._preclip_grad_dir, f'step_{step:06d}_rank_{rank:05d}')
        if os.path.exists(os.path.join(step_dir, 'manifest.json')):
            raise FileExistsError(f'Pre-clip gradient capture already exists: {step_dir}')
        os.makedirs(step_dir, exist_ok=True)

        parameters = []
        seen_parameters = set()
        parameter_index = 0
        for model_index, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                gradient = getattr(parameter, 'main_grad', None)
                source = 'main_grad'
                if gradient is None:
                    gradient = parameter.grad
                    source = 'grad'
                entry = {
                    'name': f'model{model_index}.{name}',
                    'model_index': model_index,
                    'shape': list(parameter.shape),
                    'parameter_dtype': str(parameter.dtype),
                    'gradient_present': gradient is not None,
                    'gradient_source': source if gradient is not None else None,
                    'gradient_dtype': str(gradient.dtype) if gradient is not None else None,
                    'gradient_shape': list(gradient.shape) if gradient is not None else None,
                    'numel': parameter.numel(),
                    'parts': [],
                }
                if gradient is not None:
                    if gradient.numel() != parameter.numel():
                        raise ValueError(
                            f'Gradient buffer size differs from parameter {entry["name"]}: '
                            f'gradient={gradient.numel()}, parameter={parameter.numel()}')
                    flat_gradient = gradient.detach().reshape(-1)
                    for part_index, start in enumerate(
                            range(0, flat_gradient.numel(), self._preclip_grad_chunk_numel)):
                        end = min(start + self._preclip_grad_chunk_numel, flat_gradient.numel())
                        filename = f'param_{parameter_index:06d}_part_{part_index:05d}.pt'
                        value = flat_gradient[start:end].cpu().contiguous()
                        torch.save(value, os.path.join(step_dir, filename))
                        entry['parts'].append({'file': filename, 'start': start, 'end': end})
                parameters.append(entry)
                parameter_index += 1

        manifest = {
            'format': 'swift_gkd_preclip_grad_v1',
            'tag': self._preclip_grad_tag,
            'step': step,
            'rank': rank,
            'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            'tensor_model_parallel_size': getattr(self.args, 'tensor_model_parallel_size', 1),
            'pipeline_model_parallel_size': getattr(self.args, 'pipeline_model_parallel_size', 1),
            'context_parallel_size': getattr(self.args, 'context_parallel_size', 1),
            'data_parallel_size': (
                mpu.get_data_parallel_world_size() if torch.distributed.is_initialized() else 1),
            'main_grads_dtype': str(getattr(self.args, 'main_grads_dtype', None)),
            'clip_grad': float(getattr(self.args, 'clip_grad', 0.0)),
            'chunk_numel': self._preclip_grad_chunk_numel,
            'parameter_count': len(parameters),
            'parameters': parameters,
        }
        manifest_path = os.path.join(step_dir, 'manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as file:
            json.dump(manifest, file, indent=2)
        logger.info(
            f'GKD pre-clip gradients saved: step={step}, rank={rank}, '
            f'parameters={len(parameters)}, path={step_dir}')

    def _before_optimizer_step(self):
        step = int(self.state.iteration)
        self._dtype_audit_gradients(step)
        self._dtype_audit_optimizer()
        self._validate_strict_fp32_gradients()
        if step == int(os.getenv('SWIFT_GKD_RUNTIME_AUDIT_STEP', '0')):
            update_runtime_audit_grad_dtypes(self, self.wrapped_models[0])
        if step in self._preclip_grad_steps and self._is_preclip_grad_capture_rank():
            self._capture_preclip_gradients(step)

    def _alignment_debug_active(self, step=None):
        step = int(self.state.iteration) if step is None else int(step)
        return (
            self._alignment_debug_steps > 0
            and self._alignment_debug_start_step <= step < self._alignment_debug_steps
            and self._is_debug_rank()
        )

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

    @property
    def _dlogits_isolation_input_path(self):
        return os.path.join(self._dlogits_isolation_dir, 'common_dlogits.pt')

    @property
    def _dlogits_isolation_output_path(self):
        tag = self._dlogits_isolation_tag or 'result'
        return os.path.join(self._dlogits_isolation_dir, f'full_backward_{tag}.pt')

    def _is_dlogits_isolation_target(self, step, micro_batch):
        return (
            bool(self._dlogits_isolation_mode)
            and step == self._dlogits_isolation_step
            and micro_batch == self._dlogits_isolation_micro_batch
        )

    def _capture_dlogits_provenance(self, data, labels, teacher_output, step, micro_batch):
        if not self._is_dlogits_isolation_target(step, micro_batch):
            return
        if self._dlogits_provenance is not None:
            raise RuntimeError('Common dLogits provenance was captured more than once.')
        args = self.args
        self._dlogits_provenance = {
            'input_ids': self._tensor_identity(data.get('input_ids')),
            'position_ids': self._tensor_identity(data.get('position_ids')),
            'labels': self._tensor_identity(labels),
            'num_valid': int((labels != -100).sum().item()) if labels is not None else None,
            'teacher_logits': self._tensor_identity(teacher_output.full_logits),
            'teacher_topk_logprobs': self._tensor_identity(teacher_output.topk_logprobs),
            'teacher_topk_indices': self._tensor_identity(teacher_output.topk_indices),
            'teacher_labels': self._tensor_identity(teacher_output.opsd_teacher_labels),
            'runtime': {
                key: getattr(args, key, None)
                for key in (
                    'tensor_model_parallel_size', 'pipeline_model_parallel_size',
                    'context_parallel_size', 'micro_batch_size', 'global_batch_size',
                    'padding_free', 'sequence_parallel', 'attention_backend', 'torch_dtype',
                )
            },
            'checkpoint': {
                key: str(getattr(args, key, None))
                for key in ('model', 'load', 'finetune', 'no_load_optim', 'no_load_rng', 'seed', 'data_seed')
            },
            'student_parameter_probes': self._model_parameter_probe_summary(
                self.unwrapped_models),
        }

    def _register_dlogits_isolation_hook(self, output_tensor, step, micro_batch):
        if not self._is_dlogits_isolation_target(step, micro_batch) or self._dlogits_hook_done:
            return
        if not output_tensor.requires_grad:
            raise ValueError('Common dLogits isolation requires differentiable student logits.')

        common_gradient = None
        if self._dlogits_isolation_mode == 'replay':
            if not os.path.isfile(self._dlogits_isolation_input_path):
                raise FileNotFoundError(f'Common dLogits not found: {self._dlogits_isolation_input_path}')
            payload = torch.load(self._dlogits_isolation_input_path, map_location='cpu', weights_only=True)
            common_gradient = payload['dlogits']
            if tuple(common_gradient.shape) != tuple(output_tensor.shape):
                raise ValueError(
                    f'Common dLogits shape {tuple(common_gradient.shape)} does not match '
                    f'student logits shape {tuple(output_tensor.shape)}.')

        def hook(gradient):
            if self._dlogits_isolation_mode == 'capture':
                os.makedirs(self._dlogits_isolation_dir, exist_ok=True)
                torch.save({
                    'dlogits': gradient.detach().cpu(),
                    'identity': self._tensor_identity(gradient),
                    'step': step,
                    'micro_batch': micro_batch,
                }, self._dlogits_isolation_input_path)
                logger.info(f'Captured common student dLogits: {self._dlogits_isolation_input_path}')
                replacement = gradient
            else:
                replacement = common_gradient.to(device=gradient.device, dtype=gradient.dtype)
                logger.info(f'Replaying common student dLogits: {self._dlogits_isolation_input_path}')
            self._dlogits_hook_done = True
            return replacement

        output_tensor.register_hook(hook)

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
        # Aggregate over valid tokens so GPU/NPU logs remain comparable even
        # when the first valid position or sequence lengths differ.
        if labels is not None:
            valid = labels != -100
            if valid.shape[:2] == logits.shape[:2] and valid.any():
                valid_logits = logits[valid].detach().float()
                flat_labels = labels[valid].detach().long().clamp_(0, vocab_size - 1)
                valid_log_probs = torch.log_softmax(valid_logits, dim=-1)
                target_log_probs = valid_log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
                result.update({
                    'valid_token_count': int(valid_logits.shape[0]),
                    'mean': valid_logits.mean().item(),
                    'std': valid_logits.std(unbiased=False).item(),
                    'rms': valid_logits.square().mean().sqrt().item(),
                    'min': valid_logits.min().item(),
                    'max': valid_logits.max().item(),
                    'target_logprob_mean': target_log_probs.mean().item(),
                    'target_logprob_std': target_log_probs.std(unbiased=False).item(),
                    'top1_target_agreement': (
                        valid_logits.argmax(dim=-1) == flat_labels).float().mean().item(),
                })
        return result

    def _first_layer0_parameter_mean(self):
        """Return the mean of the first Layer 0 parameter in model order."""
        for model in self.unwrapped_models or []:
            for name, parameter in model.named_parameters():
                if name.startswith('decoder.layers.0.'):
                    return name, parameter.detach().float().mean().item()
        return None, None

    def _layer0_attention_weight_mean(self):
        """Return the element-weighted mean of Layer 0 attention weights."""
        names = []
        total = None
        count = 0
        prefix = 'decoder.layers.0.self_attention.'
        for model in self.unwrapped_models or []:
            for name, parameter in model.named_parameters():
                if name.startswith(prefix) and name.endswith('.weight'):
                    value = parameter.detach().float()
                    total = value.sum() if total is None else total + value.sum()
                    count += value.numel()
                    names.append(name)
        if total is None or count == 0:
            return names, None
        return names, (total / count).item()

    def _layer0_attention_weight_stats(self):
        """Return aggregate statistics for Layer 0 attention weight tensors."""
        values = []
        names = []
        for model in self.unwrapped_models or []:
            for name, parameter in model.named_parameters():
                if name.startswith('decoder.layers.0.self_attention.') and name.endswith('.weight'):
                    values.append(parameter.detach().float().reshape(-1))
                    names.append(name)
        if not values:
            return names, None
        value = torch.cat(values)
        finite = torch.isfinite(value)
        finite_value = value[finite]
        if finite_value.numel() == 0:
            return names, {
                'norm': None, 'abs_mean': None, 'std': None, 'rms': None,
                'finite_count': 0,
            }
        return names, {
            'norm': finite_value.norm().item(),
            'abs_mean': finite_value.abs().mean().item(),
            'std': finite_value.std(unbiased=False).item(),
            'rms': finite_value.square().mean().sqrt().item(),
            'finite_count': int(finite.sum().item()),
        }

    def _dtype_audit_model(self, student_output=None, teacher_logits=None):
        """Log runtime dtypes only when explicitly enabled; never changes computation."""
        if not self._dtype_audit_enabled or self._dtype_audit_model_done or not self._is_debug_rank():
            return
        self._dtype_audit_model_done = True
        logger.info(
            f'GKD dtype audit config: torch_dtype={getattr(self.args, "torch_dtype", None)}, '
            f'bf16={getattr(self.args, "bf16", None)}, '
            f'attention_softmax_in_fp32={getattr(self.args, "attention_softmax_in_fp32", None)}, '
            f'accumulate_allreduce_grads_in_fp32={getattr(self.args, "accumulate_allreduce_grads_in_fp32", None)}, '
            f'use_precision_aware_optimizer={getattr(self.args, "use_precision_aware_optimizer", None)}, '
            f'jsd_fp32={os.getenv("SWIFT_GKD_JSD_FP32", "0")}')
        logger.info(
            f'GKD dtype audit logits: student={getattr(student_output, "dtype", None)}, '
            f'teacher={getattr(teacher_logits, "dtype", None)}')
        for tag, models in (('student', self.unwrapped_models), ('teacher', self.teacher_models)):
            if not models:
                continue
            checked = 0
            for name, parameter in models[0].named_parameters():
                if any(pattern in name for pattern in (
                        'word_embeddings.weight', 'embedding',
                        'decoder.layers.0.self_attention.linear_qkv.weight',
                        'decoder.layers.0.self_attention.linear_proj.weight',
                        'output_layer.weight')):
                    logger.info(
                        f'GKD dtype audit {tag}: {name}, parameter={parameter.dtype}, '
                        f'shape={list(parameter.shape)}')
                    checked += 1
                    if checked >= 8:
                        break

    def _dtype_audit_gradients(self, step):
        if not self._dtype_audit_enabled or self._dtype_audit_grad_done or not self._is_debug_rank():
            return
        self._dtype_audit_grad_done = True
        targets = ('word_embeddings.weight', 'decoder.layers.0.self_attention.linear_qkv.weight',
                   'decoder.layers.0.self_attention.linear_proj.weight', 'output_layer.weight')
        for model_idx, model in enumerate(self.unwrapped_models or []):
            for name, parameter in model.named_parameters():
                if not any(target in name for target in targets):
                    continue
                grad = parameter.grad
                main_grad = getattr(parameter, 'main_grad', None)
                logger.info(
                    f'GKD dtype audit grad: step={step}, model={model_idx}, name={name}, '
                    f'parameter={parameter.dtype}, grad={getattr(grad, "dtype", None)}, '
                    f'main_grad={getattr(main_grad, "dtype", None)}')

    def _dtype_audit_optimizer(self):
        if not self._dtype_audit_enabled or self._dtype_audit_optimizer_done or not self._is_debug_rank():
            return
        self._dtype_audit_optimizer_done = True
        for group in self.optimizer.param_groups:
            for parameter in group.get('params', []):
                # Distributed optimizers may expose state as ProxyDict rather
                # than a normal dict; audit access must never abort training.
                try:
                    state = self.optimizer.state[parameter]
                except (KeyError, TypeError, AttributeError):
                    state = None
                def state_value(name):
                    if state is None:
                        return None
                    try:
                        return state[name]
                    except (KeyError, TypeError, AttributeError):
                        return None
                exp_avg = state_value('exp_avg')
                exp_avg_sq = state_value('exp_avg_sq')
                logger.info(
                    f'GKD dtype audit optimizer: parameter={parameter.dtype}, '
                    f'main_param={getattr(getattr(parameter, "main_param", None), "dtype", None)}, '
                    f'exp_avg={getattr(exp_avg, "dtype", None)}, '
                    f'exp_avg_sq={getattr(exp_avg_sq, "dtype", None)}, '
                    f'state_type={type(self.optimizer.state).__name__}')
                return

    def _write_alignment_record(self, record):
        if not self._is_debug_rank():
            return
        os.makedirs(self._alignment_debug_dir, exist_ok=True)
        with open(self._alignment_debug_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

    def _grad_clip_debug_active(self, step):
        return self._grad_clip_debug_start_step <= step < self._grad_clip_debug_steps

    def _write_grad_clip_record(self, step, grad_norm, update_successful):
        if not self._is_debug_rank():
            return
        clip_grad = float(self.args.clip_grad)
        grad_norm = float(grad_norm) if grad_norm is not None else None
        clipping_enabled = clip_grad > 0.0
        clipped = clipping_enabled and grad_norm is not None and grad_norm > clip_grad
        clip_coefficient = None
        if clipping_enabled and grad_norm is not None:
            clip_coefficient = min(1.0, clip_grad / (grad_norm + 1e-6))
        config_clip_grads = []
        for optimizer in self._optimizer_objects(self.optimizer):
            config = getattr(optimizer, 'config', None)
            value = getattr(config, 'clip_grad', None)
            if value is not None:
                config_clip_grads.append(float(value))
        record = {
            'record_type': 'grad_clip_step',
            'tag': self._grad_clip_debug_tag,
            'step': step,
            'optimizer_reported_grad_norm': grad_norm,
            'clip_grad': clip_grad,
            'optimizer_config_clip_grads': sorted(set(config_clip_grads)),
            'clipping_enabled': clipping_enabled,
            'clipped': clipped,
            'clip_coefficient': clip_coefficient,
            'update_successful': bool(update_successful),
            'norm_semantics': 'optimizer.step return; expected pre-clip total norm',
        }
        os.makedirs(self._grad_clip_debug_dir, exist_ok=True)
        with open(self._grad_clip_debug_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

    def _capture_jsd_isolation_inputs(self, student_logits, teacher_output, labels, step, micro_batch):
        if self._jsd_isolation_mode != 'capture' or self._jsd_isolation_done:
            return
        if step != self._jsd_isolation_step or micro_batch != self._jsd_isolation_micro_batch:
            return
        if not self._is_debug_rank():
            return
        os.makedirs(self._jsd_isolation_dir, exist_ok=True)
        path = os.path.join(
            self._jsd_isolation_dir,
            f'jsd_inputs_step_{step:06d}_micro_{micro_batch:03d}.pt')
        torch.save({
            'student_logits': student_logits.detach().cpu(),
            'teacher_logits': (
                teacher_output.full_logits.detach().cpu()
                if teacher_output.full_logits is not None else None),
            'teacher_topk_logprobs': (
                teacher_output.topk_logprobs.detach().cpu()
                if teacher_output.topk_logprobs is not None else None),
            'teacher_topk_indices': (
                teacher_output.topk_indices.detach().cpu()
                if teacher_output.topk_indices is not None else None),
            'opsd_teacher_labels': (
                teacher_output.opsd_teacher_labels.detach().cpu()
                if teacher_output.opsd_teacher_labels is not None else None),
            'labels': labels.detach().cpu(),
            'beta': float(self.beta),
            'temperature': float(self.temperature),
            'step': int(step),
            'micro_batch': int(micro_batch),
        }, path)
        self._jsd_isolation_done = True
        logger.info(f'Captured common JSD backward inputs: {path}')

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
    def _swiglu_isolation_path(self):
        return os.path.join(self._swiglu_isolation_dir, 'swiglu_fc1_output.pt')

    def _swiglu_isolation_forward_hook(self, name):

        def hook(module, inputs, output):
            context = self._operator_debug_context
            if context is None or self._swiglu_isolation_input_captured:
                return
            if (context['step'] != self._swiglu_isolation_step
                    or context['micro_batch'] != self._swiglu_isolation_micro_batch):
                return
            tensor = self._first_tensor(output)
            if tensor is None:
                raise RuntimeError(f'SwiGLU isolation target returned no tensor: {name}')
            bias = None
            if isinstance(output, (tuple, list)) and len(output) > 1 and torch.is_tensor(output[1]):
                bias = output[1]
            os.makedirs(self._swiglu_isolation_dir, exist_ok=True)
            self._swiglu_isolation_payload = {
                'target': name,
                'dout_target': self._swiglu_isolation_dout_target,
                'step': context['step'],
                'micro_batch': context['micro_batch'],
                'fc1_output': tensor.detach().contiguous().cpu(),
                'fc1_bias': bias.detach().contiguous().cpu() if bias is not None else None,
            }
            torch.save(self._swiglu_isolation_payload, self._swiglu_isolation_path)
            self._swiglu_isolation_input_captured = True
            logger.info(f'Captured common SwiGLU input: {self._swiglu_isolation_path}')

        return hook

    def _swiglu_isolation_backward_hook(self, name):

        def hook(module, grad_input, grad_output):
            context = self._backward_debug_context
            if context is None or self._swiglu_isolation_done:
                return
            if (context['step'] != self._swiglu_isolation_step
                    or context['micro_batch'] != self._swiglu_isolation_micro_batch):
                return
            if not self._swiglu_isolation_input_captured or self._swiglu_isolation_payload is None:
                raise RuntimeError('SwiGLU backward isolation did not capture linear_fc1 output first.')
            gradient = self._first_tensor(grad_input)
            if gradient is None:
                raise RuntimeError(f'SwiGLU dout target returned no input gradient: {name}')
            payload = dict(self._swiglu_isolation_payload)
            payload.update({
                'dout_target': name,
                'swiglu_dout': gradient.detach().contiguous().cpu(),
            })
            torch.save(payload, self._swiglu_isolation_path)
            self._swiglu_isolation_done = True
            logger.info(f'Captured common SwiGLU input and dout: {self._swiglu_isolation_path}')

        return hook

    def _register_swiglu_isolation_hooks(self):
        input_matched = []
        dout_matched = []
        for model in self.unwrapped_models:
            for name, module in model.named_modules():
                if name == self._swiglu_isolation_target:
                    handle = module.register_forward_hook(self._swiglu_isolation_forward_hook(name))
                    self._swiglu_isolation_handles.append(handle)
                    input_matched.append(f'{name} ({type(module).__name__})')
                if name == self._swiglu_isolation_dout_target:
                    handle = module.register_full_backward_hook(self._swiglu_isolation_backward_hook(name))
                    self._swiglu_isolation_handles.append(handle)
                    dout_matched.append(f'{name} ({type(module).__name__})')
        if not input_matched:
            raise ValueError(f'No SwiGLU isolation module matched: {self._swiglu_isolation_target}')
        if not dout_matched:
            raise ValueError(f'No SwiGLU dout module matched: {self._swiglu_isolation_dout_target}')
        logger.info(
            f'GKD SwiGLU isolation hooks registered: input={input_matched}, dout={dout_matched}')

    @property
    def _fc1_isolation_common_path(self):
        return os.path.join(self._fc1_isolation_dir, 'fc1_module_common.pt')

    @property
    def _fc1_isolation_result_path(self):
        tag = self._fc1_isolation_tag or self._fc1_isolation_mode
        return os.path.join(self._fc1_isolation_dir, f'fc1_module_{tag}.pt')

    def _is_fc1_isolation_target(self, context):
        return (
            context is not None
            and context['step'] == self._fc1_isolation_step
            and context['micro_batch'] == self._fc1_isolation_micro_batch
        )

    def _fc1_isolation_context(self):
        if self._is_fc1_isolation_target(self._operator_debug_context):
            return self._operator_debug_context
        if self._is_fc1_isolation_target(self._backward_debug_context):
            return self._backward_debug_context
        return None

    @staticmethod
    def _named_tensor_leaves(value, prefix='output'):
        if torch.is_tensor(value):
            return [(prefix, value)]
        if isinstance(value, dict):
            result = []
            for key, item in value.items():
                result.extend(MegatronGKDTrainer._named_tensor_leaves(item, f'{prefix}.{key}'))
            return result
        if isinstance(value, (tuple, list)):
            result = []
            for index, item in enumerate(value):
                result.extend(MegatronGKDTrainer._named_tensor_leaves(item, f'{prefix}.{index}'))
            return result
        return []

    @staticmethod
    def _normalize_fc1_isolation_module_config(config):
        aliases = {
            'epsilon': (
                'epsilon',
                'module.eps',
                'module.epsilon',
                'module.layernorm_epsilon',
                'module.layer_norm_epsilon',
                'config.eps',
                'config.epsilon',
                'config.layernorm_epsilon',
                'config.layer_norm_epsilon',
            ),
            'normalization': (
                'normalization',
                'module.normalization',
                'config.normalization',
            ),
            'zero_centered_gamma': (
                'zero_centered_gamma',
                'module.zero_centered_gamma',
                'config.zero_centered_gamma',
                'config.layernorm_zero_centered_gamma',
            ),
        }
        result = {}
        for canonical_name, candidate_names in aliases.items():
            for candidate_name in candidate_names:
                value = config.get(candidate_name)
                if isinstance(value, (bool, int, float, str)):
                    result[canonical_name] = value
                    break
        return result

    @classmethod
    def _fc1_isolation_module_config(cls, module):
        raw_config = {}
        attribute_names = (
            'eps',
            'epsilon',
            'layernorm_epsilon',
            'layer_norm_epsilon',
            'normalization',
            'zero_centered_gamma',
            'layernorm_zero_centered_gamma',
        )
        for prefix, source in (('module', module), ('config', getattr(module, 'config', None))):
            if source is None:
                continue
            for name in attribute_names:
                value = getattr(source, name, None)
                if isinstance(value, (bool, int, float, str)):
                    raw_config[f'{prefix}.{name}'] = value
        return cls._normalize_fc1_isolation_module_config(raw_config)

    def _load_fc1_isolation_common(self):
        if self._fc1_isolation_common is None:
            if not os.path.isfile(self._fc1_isolation_common_path):
                raise FileNotFoundError(
                    f'Common FC1 isolation payload not found: {self._fc1_isolation_common_path}')
            self._fc1_isolation_common = torch.load(
                self._fc1_isolation_common_path, map_location='cpu', weights_only=True)
        return self._fc1_isolation_common

    def _fc1_isolation_pre_hook(self, name):

        def hook(module, args, kwargs):
            context = self._fc1_isolation_context()
            if context is None:
                return args, kwargs
            input_tensor = self._first_tensor(args)
            if input_tensor is None:
                input_tensor = self._first_tensor(kwargs)
            if input_tensor is None:
                raise RuntimeError(f'FC1 isolation target received no tensor input: {name}')

            runtime_parameters = dict(module.named_parameters())
            module_type = type(module).__name__
            module_signature = str(inspect.signature(module.forward))
            module_config = self._fc1_isolation_module_config(module)
            if self._fc1_isolation_mode == 'capture':
                if self._fc1_isolation_common is None:
                    self._fc1_isolation_common = {
                        'target': name,
                        'module_type': module_type,
                        'module_signature': module_signature,
                        'module_config': module_config,
                        'step': context['step'],
                        'micro_batch': context['micro_batch'],
                        'input': input_tensor.detach().cpu().contiguous(),
                        'parameters': {
                            key: value.detach().cpu().contiguous()
                            for key, value in runtime_parameters.items()
                        },
                        'output_gradients': {},
                    }
                isolated_input = input_tensor
            else:
                common = self._load_fc1_isolation_common()
                if common.get('target') != name:
                    raise ValueError(
                        f'Common FC1 target {common.get("target")} does not match runtime target {name}.')
                if common.get('module_type') != module_type:
                    raise ValueError(
                        f'Common FC1 module type {common.get("module_type")} does not match '
                        f'runtime type {module_type}.')
                common_module_config = self._normalize_fc1_isolation_module_config(
                    common.get('module_config', {}))
                mismatched_config = {
                    key: {'common': value, 'runtime': module_config.get(key)}
                    for key, value in common_module_config.items()
                    if module_config.get(key) != value
                }
                if mismatched_config:
                    raise ValueError(
                        f'Common FC1 module config does not match runtime config: {mismatched_config}.')
                common_input = common['input']
                if tuple(common_input.shape) != tuple(input_tensor.shape):
                    raise ValueError(
                        f'Common FC1 input shape {tuple(common_input.shape)} does not match '
                        f'runtime shape {tuple(input_tensor.shape)}.')
                common_parameters = common.get('parameters', {})
                if set(common_parameters) != set(runtime_parameters):
                    raise ValueError(
                        f'FC1 parameter names differ: common={sorted(common_parameters)}, '
                        f'runtime={sorted(runtime_parameters)}.')
                with torch.no_grad():
                    for key, parameter in runtime_parameters.items():
                        common_parameter = common_parameters[key]
                        if tuple(common_parameter.shape) != tuple(parameter.shape):
                            raise ValueError(
                                f'FC1 parameter {key} shape {tuple(common_parameter.shape)} does not match '
                                f'runtime shape {tuple(parameter.shape)}.')
                        parameter.copy_(common_parameter.to(device=parameter.device, dtype=parameter.dtype))
                isolated_input = common_input.to(
                    device=input_tensor.device, dtype=input_tensor.dtype).detach().requires_grad_(True)
                args, kwargs = self._replace_first_tensor(args, kwargs, isolated_input)

            self._fc1_isolation_module = module
            self._fc1_isolation_result = {
                'mode': self._fc1_isolation_mode,
                'tag': self._fc1_isolation_tag,
                'target': name,
                'module_type': module_type,
                'module_signature': module_signature,
                'module_config': module_config,
                'step': context['step'],
                'micro_batch': context['micro_batch'],
                'input_identity': self._tensor_identity(isolated_input),
                'parameter_identities': {
                    key: self._tensor_identity(value)
                    for key, value in runtime_parameters.items()
                },
                'forward_outputs': {},
                'used_output_gradients': {},
                'input_gradient': None,
                'input_gradient_source': None,
                'parameter_gradients': {},
            }
            return args, kwargs

        return hook

    def _fc1_isolation_forward_hook(self, name):

        def hook(module, args, kwargs, output):
            context = self._fc1_isolation_context()
            if context is None:
                return
            leaves = self._named_tensor_leaves(output)
            if not leaves:
                raise RuntimeError(f'FC1 isolation target returned no tensor output: {name}')
            result = self._fc1_isolation_result
            if result is None:
                raise RuntimeError('FC1 isolation forward hook ran before its pre-hook.')
            result['forward_outputs'] = {
                path: tensor.detach().cpu().contiguous() for path, tensor in leaves
            }
            grad_leaves = {path: tensor for path, tensor in leaves if tensor.requires_grad}
            # Reentrant activation checkpointing runs the original forward under no_grad
            # and creates the differentiable outputs during backward recomputation.
            if not grad_leaves:
                self._fc1_isolation_forward_done = True
                return
            if self._fc1_isolation_mode == 'capture':
                expected = set(grad_leaves)
            else:
                expected = set(self._load_fc1_isolation_common().get('output_gradient_paths', []))
                if not expected.issubset(grad_leaves):
                    raise ValueError(
                        f'FC1 common differentiable outputs are missing at runtime: common={sorted(expected)}, '
                        f'runtime={sorted(grad_leaves)}.')
            self._fc1_isolation_expected_output_gradients = expected

            for path, tensor in grad_leaves.items():

                def output_gradient_hook(gradient, output_path=path):
                    if self._fc1_isolation_mode == 'capture':
                        replacement = gradient
                        self._fc1_isolation_common['output_gradients'][output_path] = (
                            gradient.detach().cpu().contiguous())
                    else:
                        if output_path not in expected:
                            raise RuntimeError(
                                f'FC1 runtime output {output_path} received a gradient but was unused '
                                'during common NPU capture.')
                        common_gradient = self._load_fc1_isolation_common()['output_gradients'][output_path]
                        if tuple(common_gradient.shape) != tuple(gradient.shape):
                            raise ValueError(
                                f'Common FC1 dout {output_path} shape {tuple(common_gradient.shape)} '
                                f'does not match runtime shape {tuple(gradient.shape)}.')
                        replacement = common_gradient.to(device=gradient.device, dtype=gradient.dtype)
                    self._fc1_isolation_result['used_output_gradients'][output_path] = (
                        replacement.detach().cpu().contiguous())
                    self._fc1_isolation_seen_output_gradients.add(output_path)
                    return replacement

                tensor.register_hook(output_gradient_hook)
            self._fc1_isolation_forward_done = True

        return hook

    def _fc1_isolation_backward_hook(self, name):

        def hook(module, grad_input, grad_output):
            context = self._fc1_isolation_context()
            if context is None:
                return
            input_gradient = self._first_tensor(grad_input)
            if input_gradient is None:
                raise RuntimeError(f'FC1 isolation target returned no local input gradient: {name}')
            if self._fc1_isolation_result is None:
                raise RuntimeError('FC1 isolation backward hook ran before its forward hooks.')
            self._fc1_isolation_result['input_gradient'] = input_gradient.detach().cpu().contiguous()
            self._fc1_isolation_result['input_gradient_source'] = 'module_full_backward_hook'

        return hook

    def _register_fc1_isolation_hooks(self):
        matched = []
        for model in self.unwrapped_models:
            for name, module in model.named_modules():
                if name != self._fc1_isolation_target:
                    continue
                pre_handle = module.register_forward_pre_hook(
                    self._fc1_isolation_pre_hook(name), with_kwargs=True)
                output_handle = module.register_forward_hook(
                    self._fc1_isolation_forward_hook(name), with_kwargs=True)
                backward_handle = module.register_full_backward_hook(
                    self._fc1_isolation_backward_hook(name))
                self._fc1_isolation_handles.extend([pre_handle, output_handle, backward_handle])
                matched.append(f'{name} ({type(module).__name__}, {inspect.signature(module.forward)})')
        if not matched:
            raise ValueError(f'FC1 isolation target not found: {self._fc1_isolation_target}')
        logger.info(f'GKD FC1 isolation hooks registered: {matched}')

    def _capture_fc1_isolation_parameter_gradients(self):
        result = {}
        for name, parameter in self._fc1_isolation_module.named_parameters():
            gradient = getattr(parameter, 'main_grad', None)
            if gradient is None:
                gradient = parameter.grad
            result[name] = gradient.detach().cpu().contiguous() if gradient is not None else None
        return result

    def _backward_debug_hook(self, model_idx, name):

        def hook(module, grad_input, grad_output):
            context = self._backward_debug_context
            if context is None or not self._alignment_debug_active(context['step']):
                return
            key = (context['step'], context['micro_batch'], model_idx, name)
            if key in self._backward_debug_written:
                return
            self._backward_debug_written.add(key)
            input_gradient = self._first_tensor(grad_input)
            output_gradient = self._first_tensor(grad_output)
            if self._is_dlogits_isolation_target(context['step'], context['micro_batch']):
                capture_key = f'model{model_idx}.{name}'
                self._dlogits_module_gradients[capture_key] = {
                    'module_type': type(module).__name__,
                    'input_gradient': input_gradient.detach().cpu() if input_gradient is not None else None,
                    # output_layer receives the full vocabulary-sized dLogits,
                    # which is already stored once in common_dlogits.pt.
                    'output_gradient': (
                        None if name == 'output_layer' or output_gradient is None
                        else output_gradient.detach().cpu()),
                }
            self._write_alignment_record({
                'record_type': 'module_backward',
                'step': context['step'],
                'micro_batch': context['micro_batch'],
                'name': f'model{model_idx}.{name}',
                'module_type': type(module).__name__,
                'input_gradient': self._operator_tensor_summary(input_gradient),
                'output_gradient': self._operator_tensor_summary(output_gradient),
            })

        return hook

    def _register_backward_debug_hooks(self):
        matched = []
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, module in model.named_modules():
                if name not in self._backward_debug_patterns:
                    continue
                handle = module.register_full_backward_hook(self._backward_debug_hook(model_idx, name))
                self._backward_debug_handles.append(handle)
                matched.append(f'model{model_idx}.{name} ({type(module).__name__})')
        if not matched:
            raise ValueError(
                f'No backward debug module matched exact patterns: {self._backward_debug_patterns}')
        logger.info(f'GKD backward debug hooks registered: {matched}')

    def _capture_dlogits_parameter_gradient_samples(self):
        result = {}
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                if name not in self._dlogits_parameter_patterns:
                    continue
                gradient = getattr(parameter, 'main_grad', None)
                if gradient is None:
                    gradient = parameter.grad
                if gradient is None:
                    continue
                indices = self._sample_parameter_indices(gradient.numel())
                index_tensor = torch.tensor(indices, dtype=torch.int64, device=gradient.device)
                sample = gradient.detach().reshape(-1).index_select(0, index_tensor).cpu()
                result[f'model{model_idx}.{name}'] = {
                    'full_shape': list(gradient.shape),
                    'full_dtype': str(gradient.dtype),
                    'full_numel': gradient.numel(),
                    'sample_indices': indices,
                    'sample': sample,
                }
        return result

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

    @property
    def _flash_isolation_input_path(self):
        return os.path.join(self._flash_isolation_dir, 'flash_inputs.pt')

    def _flash_isolation_output_path(self, tensor):
        tag = self._flash_isolation_tag or tensor.device.type
        return os.path.join(self._flash_isolation_dir, f'flash_output_{tag}.pt')

    def _flash_backward_isolation_output_path(self, tensor):
        tag = self._flash_isolation_tag or tensor.device.type
        return os.path.join(self._flash_isolation_dir, f'flash_backward_{tag}.pt')

    @staticmethod
    def _flash_direct_tensors(args, kwargs):
        tensors = []
        for index, value in enumerate(args):
            if torch.is_tensor(value):
                tensors.append({'location': 'arg', 'key': index, 'tensor': value})
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                tensors.append({'location': 'kwarg', 'key': key, 'tensor': value})
        return tensors

    @staticmethod
    def _flash_call_metadata(args, kwargs):
        return {
            'arg_count': len(args),
            'kwarg_keys': sorted(kwargs),
            'non_tensor_args': {
                str(index): repr(value) for index, value in enumerate(args) if not torch.is_tensor(value)
            },
            'non_tensor_kwargs': {
                key: repr(value) for key, value in kwargs.items() if not torch.is_tensor(value)
            },
        }

    @staticmethod
    def _flash_comparable_metadata(metadata):
        metadata = dict(metadata)
        non_tensor_kwargs = dict(metadata.get('non_tensor_kwargs', {}))

        # PackedSeqParams contains backend-specific devices and optional padded
        # sequence tensors. Replay must keep the runtime backend's own object.
        non_tensor_kwargs.pop('packed_seq_params', None)
        metadata['non_tensor_kwargs'] = non_tensor_kwargs
        return metadata

    def _run_flash_backward_isolation(self, module, args, kwargs):
        if not self._flash_backward_isolation_enabled or self._flash_backward_isolation_done:
            return
        if len(args) < 3 or not all(torch.is_tensor(args[index]) for index in range(3)):
            raise ValueError('Flash backward isolation requires Q/K/V as the first three positional tensors.')

        isolated_args = list(args)
        qkv = []
        for index in range(3):
            value = args[index].detach().clone().requires_grad_(True)
            isolated_args[index] = value
            qkv.append(value)

        # This direct call bypasses this module's hooks and avoids activation
        # checkpoint recomputation changing the common Q/K/V used by the probe.
        with torch.enable_grad():
            isolated_output = module.forward(*isolated_args, **kwargs)
            output_tensor = self._first_tensor(isolated_output)
            if output_tensor is None or not output_tensor.requires_grad:
                raise ValueError('Flash backward isolation did not produce a differentiable output tensor.')

            # Generate the same flattened gradient on CPU. The backend-specific
            # output layouts may differ while retaining the same element order.
            common_dout = torch.linspace(
                -1.0, 1.0, output_tensor.numel(), dtype=torch.float32, device='cpu')
            common_dout = common_dout.to(dtype=output_tensor.dtype)
            common_dout = common_dout.reshape(output_tensor.shape).to(device=output_tensor.device)
            gradients = torch.autograd.grad(
                outputs=output_tensor,
                inputs=tuple(qkv),
                grad_outputs=common_dout,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

        output_path = self._flash_backward_isolation_output_path(output_tensor)
        torch.save({
            'dq': gradients[0].detach().cpu(),
            'dk': gradients[1].detach().cpu(),
            'dv': gradients[2].detach().cpu(),
            'qkv': [self._tensor_identity(value) for value in qkv],
            'dout': self._tensor_identity(common_dout),
            'output_shape': list(output_tensor.shape),
            'output_dtype': str(output_tensor.dtype),
            'mode': self._flash_isolation_mode,
            'target_prefix': self._flash_isolation_prefix,
            'module_type': type(module).__name__,
        }, output_path)
        self._flash_backward_isolation_done = True
        logger.info(f'Saved isolated Flash Attention backward gradients: {output_path}')

    def _flash_isolation_pre_hook(self, module, args, kwargs):
        context = self._operator_debug_context
        if context is None or (
                context['step'] != self._flash_isolation_step
                or context['micro_batch'] != self._flash_isolation_micro_batch):
            return args, kwargs
        if self._flash_isolation_done:
            return args, kwargs

        runtime_tensors = self._flash_direct_tensors(args, kwargs)
        if len(runtime_tensors) < 3:
            raise ValueError(
                f'Flash Attention isolation expected at least Q/K/V tensors, found {len(runtime_tensors)}.')
        os.makedirs(self._flash_isolation_dir, exist_ok=True)

        if self._flash_isolation_mode == 'capture':
            payload_tensors = []
            for item in runtime_tensors:
                payload_tensors.append({
                    'location': item['location'],
                    'key': item['key'],
                    'tensor': item['tensor'].detach().cpu(),
                    'shape': list(item['tensor'].shape),
                    'dtype': str(item['tensor'].dtype),
                })
            torch.save({
                'tensors': payload_tensors,
                'metadata': self._flash_call_metadata(args, kwargs),
                'target_prefix': self._flash_isolation_prefix,
                'module_type': type(module).__name__,
            }, self._flash_isolation_input_path)
            logger.info(f'Captured common Flash Attention inputs: {self._flash_isolation_input_path}')
            self._run_flash_backward_isolation(module, args, kwargs)
            return args, kwargs

        if not os.path.isfile(self._flash_isolation_input_path):
            raise FileNotFoundError(f'Common Flash Attention inputs not found: {self._flash_isolation_input_path}')
        payload = torch.load(self._flash_isolation_input_path, map_location='cpu', weights_only=True)
        saved_tensors = payload['tensors']
        saved_metadata = payload['metadata']
        runtime_metadata = self._flash_call_metadata(args, kwargs)
        saved_comparable = self._flash_comparable_metadata(saved_metadata)
        runtime_comparable = self._flash_comparable_metadata(runtime_metadata)
        if saved_comparable != runtime_comparable:
            raise ValueError(
                f'Flash Attention call metadata does not match the captured call. '
                f'captured={saved_comparable}, runtime={runtime_comparable}')
        runtime_by_location = {
            (item['location'], item['key']): item for item in runtime_tensors
        }
        args = list(args)
        kwargs = dict(kwargs)
        for saved in saved_tensors:
            location_key = (saved['location'], saved['key'])
            if location_key not in runtime_by_location:
                raise ValueError(f'Flash Attention runtime argument is missing: {location_key}.')
            runtime = runtime_by_location[location_key]['tensor']
            common = saved['tensor']
            if tuple(common.shape) != tuple(runtime.shape):
                raise ValueError(
                    f'Flash Attention tensor {location_key} shape {tuple(common.shape)} does not match '
                    f'runtime shape {tuple(runtime.shape)}.')
            common = common.to(device=runtime.device, dtype=runtime.dtype)
            if saved['location'] == 'arg':
                args[int(saved['key'])] = common
            else:
                kwargs[saved['key']] = common
        self._run_flash_backward_isolation(module, tuple(args), kwargs)
        logger.info(f'Replaying common Flash Attention inputs: {self._flash_isolation_input_path}')
        return tuple(args), kwargs

    def _flash_isolation_forward_hook(self, module, args, kwargs, output):
        context = self._operator_debug_context
        if context is None or (
                context['step'] != self._flash_isolation_step
                or context['micro_batch'] != self._flash_isolation_micro_batch):
            return
        if self._flash_isolation_done:
            return
        output_tensor = self._first_tensor(output)
        if output_tensor is None:
            raise ValueError('Unable to capture the Flash Attention output tensor.')
        runtime_tensors = self._flash_direct_tensors(args, kwargs)
        if len(runtime_tensors) < 3:
            raise ValueError(
                f'Flash Attention isolation expected Q/K/V tensors at output capture, '
                f'found {len(runtime_tensors)}.')
        output_path = self._flash_isolation_output_path(output_tensor)
        torch.save({
            'output': output_tensor.detach().cpu(),
            'shape': list(output_tensor.shape),
            'dtype': str(output_tensor.dtype),
            'mode': self._flash_isolation_mode,
            'target_prefix': self._flash_isolation_prefix,
            'module_type': type(module).__name__,
            'qkv': [self._tensor_identity(item['tensor']) for item in runtime_tensors[:3]],
        }, output_path)
        self._flash_isolation_done = True
        logger.info(f'Saved isolated Flash Attention output: {output_path}')

    def _register_flash_isolation_hooks(self):
        matched = []
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, module in model.named_modules():
                # GPU and NPU FlashAttention children expose different call APIs.
                # Capture and replay at their common DotProductAttention parent.
                if name != self._flash_isolation_prefix:
                    continue
                if type(module).__name__ not in {'TEDotProductAttention', 'DotProductAttention'}:
                    continue
                pre_handle = module.register_forward_pre_hook(
                    self._flash_isolation_pre_hook, with_kwargs=True)
                output_handle = module.register_forward_hook(
                    self._flash_isolation_forward_hook, with_kwargs=True)
                self._flash_isolation_handles.extend([pre_handle, output_handle])
                matched.append(f'model{model_idx}.{name} ({type(module).__name__})')
        if len(matched) != 1:
            raise ValueError(
                f'Expected exactly one DotProductAttention isolation target at '
                f'{self._flash_isolation_prefix}, found: {matched}')
        logger.info(f'GKD Flash Attention isolation mode={self._flash_isolation_mode}, targets={matched}')

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

    def _backward_debug_parameters(self):
        result = []
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, parameter in model.named_parameters():
                module_name = name.rsplit('.', 1)[0]
                if parameter.requires_grad and module_name in self._backward_debug_patterns:
                    result.append((f'model{model_idx}.{name}', parameter))
        return result

    @staticmethod
    def _optimizer_objects(optimizer):
        pending = [optimizer]
        visited = set()
        while pending:
            current = pending.pop()
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            yield current
            child = getattr(current, 'optimizer', None)
            if child is not None and child is not current:
                pending.append(child)
            pending.extend(getattr(current, 'chained_optimizers', None) or [])

    def _optimizer_parameter_state(self, parameter):
        candidates = [parameter]
        main_param = getattr(parameter, 'main_param', None)
        if torch.is_tensor(main_param):
            candidates.append(main_param)
        optimizer_type = type(self.optimizer).__name__
        for optimizer in self._optimizer_objects(self.optimizer):
            optimizer_type = f'{optimizer_type}->{type(optimizer).__name__}'
            fp16_groups = getattr(optimizer, 'float16_groups', None) or []
            fp32_groups = getattr(optimizer, 'fp32_from_float16_groups', None) or []
            for model_group, master_group in zip(fp16_groups, fp32_groups):
                for model_parameter, master_parameter in zip(model_group, master_group):
                    if model_parameter is parameter:
                        candidates.append(master_parameter)
            state = getattr(optimizer, 'state', None)
            if state is None:
                continue
            for candidate in candidates:
                if candidate not in state:
                    continue
                values = state[candidate]
                master = candidate if candidate is not parameter else main_param
                return {
                    'optimizer_type': optimizer_type,
                    'master_parameter': self._operator_tensor_summary(master) if torch.is_tensor(master) else None,
                    'exp_avg': self._operator_tensor_summary(values.get('exp_avg')),
                    'exp_avg_sq': self._operator_tensor_summary(values.get('exp_avg_sq')),
                    'step': str(values.get('step')) if values.get('step') is not None else None,
                }
        return {
            'optimizer_type': optimizer_type,
            'master_parameter': self._operator_tensor_summary(main_param) if torch.is_tensor(main_param) else None,
            'exp_avg': None,
            'exp_avg_sq': None,
            'step': None,
        }

    def _capture_optimizer_debug_state(self):
        return {
            name: self._optimizer_parameter_state(parameter)
            for name, parameter in self._backward_debug_parameters()
        }

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
        self._microbatch_backward_trace.prepare_train_step(step, debug_active)
        if self._dlogits_isolation_mode and step == self._dlogits_isolation_step:
            if self.args.num_microbatches != 1:
                raise ValueError(
                    'Common dLogits full-backward isolation requires exactly one micro-batch. '
                    'Set global_batch_size equal to micro_batch_size for this diagnostic run.')
            if not debug_active:
                raise ValueError(
                    'Common dLogits isolation target step must be inside the alignment debug window.')
        if self._fc1_isolation_mode and step == self._fc1_isolation_step:
            if self.args.num_microbatches != 1:
                raise ValueError(
                    'FC1 isolation requires exactly one micro-batch. '
                    'Set global_batch_size equal to micro_batch_size for this diagnostic run.')
            if not debug_active:
                raise ValueError('FC1 isolation target step must be inside the alignment debug window.')
        self._mlp_merge_isolation.prepare_train_step(
            step, debug_active, self.args.num_microbatches)
        self._self_attention_forward_isolation.prepare_train_step(
            step, debug_active, self.args.num_microbatches)
        self._attention_forward_trace.prepare_train_step(
            step, debug_active, self.args.num_microbatches)
        before = self._capture_trainable_tensors() if debug_active else None
        optimizer_before = (
            self._capture_optimizer_debug_state()
            if debug_active and self._backward_debug_enabled else None
        )
        result = super().train_step(train_data_iterator)
        self._validate_strict_fp32_optimizer()
        self._microbatch_backward_trace.finalize_train_step(
            step, self.args.num_microbatches)
        if self._grad_clip_debug_active(step):
            _, grad_norm, update_successful = result
            self._write_grad_clip_record(step, grad_norm, update_successful)
        if self._swiglu_isolation_mode and step == self._swiglu_isolation_step:
            if not self._swiglu_isolation_done:
                raise RuntimeError(
                    'SwiGLU backward isolation completed without capturing both fc1 output and fc2 input gradient.')
        if self._fc1_isolation_mode and step == self._fc1_isolation_step:
            if not self._fc1_isolation_forward_done or self._fc1_isolation_result is None:
                raise RuntimeError('FC1 isolation target did not run at the selected step and micro-batch.')
            if not self._fc1_isolation_seen_output_gradients:
                raise RuntimeError('FC1 isolation did not observe any output gradient.')
            if (self._fc1_isolation_mode == 'replay'
                    and self._fc1_isolation_expected_output_gradients
                    != self._fc1_isolation_seen_output_gradients):
                raise RuntimeError(
                    'FC1 isolation did not observe every output gradient: '
                    f'expected={sorted(self._fc1_isolation_expected_output_gradients)}, '
                    f'seen={sorted(self._fc1_isolation_seen_output_gradients)}.')
            if self._fc1_isolation_result.get('input_gradient') is None:
                raise RuntimeError('FC1 isolation did not capture dInput.')
            self._fc1_isolation_result['parameter_gradients'] = (
                self._capture_fc1_isolation_parameter_gradients())
            if self._fc1_isolation_mode == 'capture':
                captured_paths = set(self._fc1_isolation_common.get('output_gradients', {}))
                self._fc1_isolation_expected_output_gradients = captured_paths
                self._fc1_isolation_common['output_gradient_paths'] = sorted(captured_paths)
                torch.save(self._fc1_isolation_common, self._fc1_isolation_common_path)
                logger.info(f'Saved common FC1 isolation payload: {self._fc1_isolation_common_path}')
            torch.save(self._fc1_isolation_result, self._fc1_isolation_result_path)
            logger.info(f'Saved FC1 isolation result: {self._fc1_isolation_result_path}')
        self._mlp_merge_isolation.finalize_train_step(step)
        self._self_attention_forward_isolation.finalize_train_step(step)
        self._attention_forward_trace.finalize_train_step(step)
        if self._dlogits_isolation_mode and step == self._dlogits_isolation_step:
            if not self._dlogits_hook_done:
                raise RuntimeError(
                    'Common dLogits isolation completed without capturing or replaying the target gradient.')
            torch.save({
                'mode': self._dlogits_isolation_mode,
                'tag': self._dlogits_isolation_tag,
                'step': step,
                'micro_batch': self._dlogits_isolation_micro_batch,
                'common_dlogits': torch.load(
                    self._dlogits_isolation_input_path,
                    map_location='cpu',
                    weights_only=True,
                )['identity'],
                'provenance': self._dlogits_provenance,
                'backward_patterns': list(self._dlogits_backward_patterns),
                'module_gradients': self._dlogits_module_gradients,
                'parameter_gradient_samples': self._capture_dlogits_parameter_gradient_samples(),
            }, self._dlogits_isolation_output_path)
            logger.info(f'Saved common-dLogits full backward result: {self._dlogits_isolation_output_path}')
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
                'optimizer_state_before': optimizer_before,
                'optimizer_state_after': (
                    self._capture_optimizer_debug_state() if self._backward_debug_enabled else None),
            })
        return result

    def train(self, train_dataset, val_dataset):
        if self.truncation_strategy == 'delete':
            self.resample_data_iterator = self._init_resample_data_iterator(train_dataset)
        super().train(train_dataset, val_dataset)

    def prepare_model(self):
        super().prepare_model()
        self._validate_strict_fp32_models('student', self.unwrapped_models)
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
        self._validate_strict_fp32_models('teacher', self.teacher_models)

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
                  data_source: DataSource = DataSource.DATASET,
                  logging_micro_batch: Optional[int] = None,
                  teacher_logits_mean: Optional[float] = None,
                  layer0_first_parameter_mean: Optional[float] = None,
                  layer0_attention_weight_mean: Optional[float] = None,
                  layer0_attention_weight_stats: Optional[Dict[str, Any]] = None):
        """Compute GKD loss (JSD + optional SFT loss)."""
        student_logits = output_tensor
        self._validate_strict_fp32_tensors(
            student_logits=student_logits,
            teacher_logits=teacher_output.full_logits,
            teacher_topk_logprobs=teacher_output.topk_logprobs)

        jsd_total, jsd_num_valid = gkd_loss(
            student_logits,
            teacher_output,
            labels,
            self.beta,
            self.temperature,
            gather_fn=tp_gather_topk,
            log_softmax_fn=vocab_parallel_log_softmax,
            kl_div_fn=vocab_parallel_kl_div)
        self._validate_strict_fp32_tensors(jsd_total=jsd_total)
        if self._dtype_audit_enabled and self._is_debug_rank() and not getattr(self, '_dtype_audit_jsd_done', False):
            self._dtype_audit_jsd_done = True
            logger.info(
                f'GKD dtype audit JSD: student_input={student_logits.dtype}, '
                f'teacher_input={getattr(teacher_output.full_logits, "dtype", None)}, '
                f'jsd_total={jsd_total.dtype}, '
                f'jsd_fp32={os.getenv("SWIFT_GKD_JSD_FP32", "0")}')
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
            self._validate_strict_fp32_tensors(sft_per_token_loss=per_token_loss)
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

        self._validate_strict_fp32_tensors(gkd_loss=loss)

        metric = {'loss': loss.detach().clone()}
        if logging_micro_batch is not None:
            # These scalar metrics are aggregated into the standard Megatron
            # logging.jsonl record together with loss.
            if teacher_logits_mean is not None:
                metric[f'micro{logging_micro_batch}_teacher_logits_mean'] = output_tensor.new_tensor(
                    teacher_logits_mean).detach()
            if layer0_first_parameter_mean is not None:
                metric['layer0_first_parameter_mean'] = output_tensor.new_tensor(
                    layer0_first_parameter_mean).detach()
            if layer0_attention_weight_mean is not None:
                metric['layer0_attention_weight_mean'] = output_tensor.new_tensor(
                    layer0_attention_weight_mean).detach()
            for stat_name in ('norm', 'abs_mean', 'std', 'rms', 'finite_count'):
                stat_value = (layer0_attention_weight_stats or {}).get(stat_name)
                if stat_value is not None:
                    metric[f'layer0_attention_weight_{stat_name}'] = output_tensor.new_tensor(
                        stat_value).detach()
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
        if self._flash_isolation_mode and self._is_debug_rank():
            logger.info(
                f'GKD Flash Attention isolation forward context: step={step}, '
                f'micro_batch={micro_idx}, debug_active={debug_active}')
        if debug_active:
            self._alignment_loss_context = {}

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        if debug_active and (
                self._operator_debug_enabled
                or self._linear_proj_isolation_mode
                or self._flash_isolation_mode
                or self._swiglu_isolation_mode
                or self._fc1_isolation_mode
                or self._mlp_merge_isolation.enabled
                or self._self_attention_forward_isolation.enabled
                or self._attention_forward_trace.enabled):
            self._operator_debug_context = {
                'step': step,
                'micro_batch': micro_idx,
                'call_counts': {},
            }
        self._attention_forward_trace.capture_provenance(
            data, labels, teacher_output, step, micro_idx)
        self._capture_dlogits_provenance(
            data, labels, teacher_output, step, micro_idx)
        try:
            student_output = model(**data)
        finally:
            self._operator_debug_context = None
        self._validate_strict_fp32_tensors(
            student_logits=student_output,
            teacher_logits=teacher_output.full_logits,
            teacher_topk_logprobs=teacher_output.topk_logprobs)
        write_runtime_audit(self, model, data, labels, teacher_output.full_logits, step, micro_idx)
        self._dtype_audit_model(student_output, teacher_output.full_logits)
        if (self._flash_isolation_mode
                and step == self._flash_isolation_step
                and micro_idx == self._flash_isolation_micro_batch
                and not self._flash_isolation_done):
            raise RuntimeError(
                'Flash Attention isolation target forward completed without capture. '
                f'target={self._flash_isolation_prefix}, step={step}, micro_batch={micro_idx}.')

        self._register_dlogits_isolation_hook(student_output, step, micro_idx)

        self._capture_jsd_isolation_inputs(
            student_output, teacher_output, labels, step, micro_idx)
        self._microbatch_backward_trace.capture_forward(
            data, labels, teacher_output, student_output, step, micro_idx)

        teacher_logits = teacher_output.full_logits
        teacher_logits_mean = (
            teacher_logits.detach().float().mean().item() if teacher_logits is not None else None)
        layer0_first_parameter_name, layer0_first_parameter_mean = (
            self._first_layer0_parameter_mean())
        layer0_attention_weight_names, layer0_attention_weight_mean = (
            self._layer0_attention_weight_mean())
        _, layer0_attention_weight_stats = self._layer0_attention_weight_stats()
        if debug_active:
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
                # Full-vocabulary teacher logits mean for this exact step/micro.
                # This is None when top-k teacher output is configured.
                'teacher_logits_mean': teacher_logits_mean,
                'layer0_first_parameter_name': layer0_first_parameter_name,
                'layer0_first_parameter_mean': layer0_first_parameter_mean,
                'layer0_attention_weight_names': layer0_attention_weight_names,
                'layer0_attention_weight_mean': layer0_attention_weight_mean,
                'layer0_attention_weight_stats': layer0_attention_weight_stats,
            }
            if teacher_logits is None:
                record['teacher_topk_logprobs'] = self._tensor_summary(teacher_output.topk_logprobs)
                record['teacher_topk_indices'] = self._tensor_identity(teacher_output.topk_indices)

            def write_loss_record():
                record['loss'] = self._alignment_loss_context
                loss_context = self._alignment_loss_context or {}
                loss_value = loss_context.get('loss', loss_context.get('jsd_loss'))
                logger.info(
                    f'GKD teacher/logging: step={step}, micro_batch={micro_idx}, '
                    f'loss={loss_value}, teacher_logits_mean={teacher_logits_mean}, '
                    f'layer0_first_parameter={layer0_first_parameter_name}, '
                    f'layer0_first_parameter_mean={layer0_first_parameter_mean}, '
                    f'layer0_attention_weight_mean={layer0_attention_weight_mean}, '
                    f'layer0_attention_weight_stats={layer0_attention_weight_stats}')
                self._write_alignment_record(record)
                self._alignment_loss_context = None

            loss_callback = partial(
                self.loss_func,
                labels=labels,
                teacher_output=teacher_output,
                data_source=data_source,
                logging_micro_batch=micro_idx,
                teacher_logits_mean=teacher_logits_mean,
                layer0_first_parameter_mean=layer0_first_parameter_mean,
                layer0_attention_weight_stats=layer0_attention_weight_stats,
                layer0_attention_weight_mean=layer0_attention_weight_mean,
            )

            def debug_loss_callback(output_tensor):
                if (self._backward_debug_enabled
                        or self._dlogits_isolation_mode
                        or self._swiglu_isolation_mode
                        or self._fc1_isolation_mode
                        or self._mlp_merge_isolation.enabled
                        or self._microbatch_backward_trace.enabled):
                    self._backward_debug_context = {
                        'step': step,
                        'micro_batch': micro_idx,
                    }

                    def logits_gradient_hook(gradient):
                        self._write_alignment_record({
                            'record_type': 'student_logits_backward',
                            'step': step,
                            'micro_batch': micro_idx,
                            'gradient': self._operator_tensor_summary(gradient),
                        })
                        return gradient

                    if output_tensor.requires_grad:
                        output_tensor.register_hook(logits_gradient_hook)
                result = loss_callback(output_tensor)
                self._microbatch_backward_trace.capture_loss(
                    step,
                    micro_idx,
                    result[0],
                    result[1],
                    self._alignment_loss_context,
                )
                write_loss_record()
                return result

            return student_output, debug_loss_callback

        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_output=teacher_output,
            data_source=data_source,
            logging_micro_batch=micro_idx,
            teacher_logits_mean=teacher_logits_mean,
            layer0_first_parameter_mean=layer0_first_parameter_mean,
            layer0_attention_weight_stats=layer0_attention_weight_stats,
            layer0_attention_weight_mean=layer0_attention_weight_mean,
        )
