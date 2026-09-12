import hashlib
import inspect
import json
import os
import random
from pathlib import Path

import torch


def _digest(value, stats=False):
    if value is None:
        return None
    if not torch.is_tensor(value):
        return value
    cpu = value.detach().contiguous().cpu()
    hash_value = cpu.float() if cpu.dtype == torch.bfloat16 else cpu
    raw = hash_value.numpy().tobytes()
    result = {
        'shape': list(cpu.shape),
        'dtype': str(value.dtype),
        'sha256': hashlib.sha256(raw).hexdigest(),
    }
    if stats and cpu.numel():
        x = cpu.float()
        result.update({'min': float(x.min()), 'max': float(x.max()),
                       'mean': float(x.mean()), 'std': float(x.std()),
                       'norm': float(torch.linalg.vector_norm(x))})
    return result


def _rng_state():
    result = {'torch_initial_seed': int(torch.initial_seed())}
    result['python_state_prefix'] = list(random.getstate()[1][:5])
    result['torch_cpu'] = _digest(torch.get_rng_state())
    if torch.cuda.is_available():
        result['cuda'] = _digest(torch.cuda.get_rng_state())
    if hasattr(torch, 'npu') and torch.npu.is_available():
        try:
            result['npu'] = _digest(torch.npu.get_rng_state())
        except Exception as exc:
            result['npu_error'] = repr(exc)
    return result


def _module_info(module):
    try:
        signature = str(inspect.signature(module.forward))
    except Exception:
        signature = None
    return {'type': type(module).__name__, 'module': type(module).__module__,
            'signature': signature}


def write_runtime_audit(trainer, model, data, labels, teacher_logits, step, micro_idx):
    """Write one compact, rank-0 runtime audit JSON per process."""
    if getattr(trainer, '_runtime_audit_written', False):
        return
    if step != int(os.getenv('SWIFT_GKD_RUNTIME_AUDIT_STEP', '0')):
        return
    if micro_idx != int(os.getenv('SWIFT_GKD_RUNTIME_AUDIT_MICRO_BATCH', '0')):
        return
    if hasattr(trainer, '_is_debug_rank') and not trainer._is_debug_rank():
        return
    output = os.getenv('SWIFT_GKD_RUNTIME_AUDIT_PATH')
    if not output:
        return

    args = trainer.args
    modules = dict(model.named_modules())
    route_names = [
        'decoder.layers.0.self_attention',
        'decoder.layers.0.self_attention.core_attention',
        'decoder.layers.0.self_attention.linear_qkv',
        'decoder.layers.0.self_attention.linear_proj',
        'decoder.layers.0.mlp.linear_fc1',
        'decoder.layers.0.mlp.linear_fc2',
    ]
    routes = {name: _module_info(modules[name]) for name in route_names if name in modules}
    packed = data.get('packed_seq_params')
    packed_info = None
    if packed is not None:
        packed_info = {}
        for name in ('qkv_format', 'max_seqlen_q', 'max_seqlen_kv',
                     'cu_seqlens_q', 'cu_seqlens_kv',
                     'cu_seqlens_q_padded', 'cu_seqlens_kv_padded'):
            value = getattr(packed, name, None)
            packed_info[name] = _digest(value) if torch.is_tensor(value) else value

    param_patterns = [p.strip() for p in os.getenv(
        'SWIFT_GKD_RUNTIME_AUDIT_PARAMETER_PATTERNS',
        'word_embeddings.weight,output_layer.weight,decoder.layers.0.self_attention.linear_qkv.weight,decoder.layers.27.mlp.linear_fc2.weight').split(',') if p.strip()]
    parameters = {}
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in param_patterns):
            parameters[name] = _digest(param, stats=True)

    ddp = getattr(model, 'ddp_config', None)
    if ddp is None:
        ddp = getattr(getattr(model, 'module', None), 'ddp_config', None)
    if ddp is None:
        ddp = getattr(trainer, 'ddp_config', None)
    grad_dtypes = {}
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in param_patterns):
            grad = getattr(param, 'main_grad', None)
            if grad is None:
                grad = param.grad
            grad_dtypes[name] = str(grad.dtype) if grad is not None else None
    audit = {
        'step': step, 'micro_batch': micro_idx,
        'device': str(next(model.parameters()).device),
        'model_training': bool(model.training),
        'module_routes': routes,
        'args': {key: str(getattr(args, key, None)) for key in (
            'torch_dtype', 'fp16', 'bf16', 'fp8_format',
            'attention_backend', 'use_flash_attn', 'padding_free', 'sequence_parallel',
            'attention_softmax_in_fp32', 'apply_query_key_layer_scaling',
            'recompute_granularity', 'recompute_modules', 'bias_dropout_fusion',
            'bias_activation_fusion', 'gradient_accumulation_fusion',
            'cross_entropy_loss_fusion', 'overlap_grad_reduce', 'align_grad_reduce',
            'micro_batch_size', 'global_batch_size', 'data_parallel_size',
            'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype',
            'accumulate_allreduce_grads_in_fp32', 'use_precision_aware_optimizer')},
        'model_config': {key: str(getattr(trainer.config, key, None)) for key in (
            'params_dtype', 'attention_backend', 'use_flash_attn', 'fp32_residual_connection')},
        'strict_fp32': os.getenv('SWIFT_GKD_STRICT_FP32', '0'),
        'jsd_fp32': os.getenv('SWIFT_GKD_JSD_FP32', '0'),
        'flash_bf16': os.getenv('SWIFT_GKD_FLASH_BF16', '0'),
        'effective_ddp': {key: str(getattr(ddp, key, None)) for key in (
            'grad_reduce_in_fp32', 'overlap_grad_reduce', 'align_grad_reduce')},
        'effective_main_grad_dtypes': grad_dtypes,
        'inputs': {key: _digest(data.get(key), stats=False) for key in ('input_ids', 'position_ids')},
        'labels': _digest(labels),
        'num_valid': int((labels != -100).sum().item()) if labels is not None else None,
        'attention_mask': _digest(data.get('attention_mask')),
        'packed_seq_params': packed_info,
        'teacher_logits': _digest(teacher_logits, stats=True),
        'parameters': parameters,
        'rng': _rng_state(),
        'accumulation': {'micro_batch_size': getattr(args, 'micro_batch_size', None),
                         'global_batch_size': getattr(args, 'global_batch_size', None),
                         'data_parallel_size': getattr(args, 'data_parallel_size', None)},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding='utf-8')
    trainer._runtime_audit_written = True


def update_runtime_audit_grad_dtypes(trainer, model):
    """Add pre-optimizer gradient dtypes to the one-shot audit file."""
    output = os.getenv('SWIFT_GKD_RUNTIME_AUDIT_PATH')
    if not output or not getattr(trainer, '_runtime_audit_written', False):
        return
    path = Path(output)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding='utf-8'))
    patterns = [p.strip() for p in os.getenv(
        'SWIFT_GKD_RUNTIME_AUDIT_PARAMETER_PATTERNS', '').split(',') if p.strip()]
    dtypes = {}
    for name, param in model.named_parameters():
        if patterns and not any(pattern in name for pattern in patterns):
            continue
        grad = getattr(param, 'main_grad', None)
        if grad is None:
            grad = param.grad
        dtypes[name] = str(grad.dtype) if grad is not None else None
    payload['pre_optimizer_grad_dtypes'] = dtypes
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
