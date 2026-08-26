# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re

import torch

from swift.utils import get_logger


logger = get_logger()


class GKDMicrobatchBackwardTrace:
    """Capture native logits, loss, dLogits, and backward boundaries per micro-batch."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.enabled = os.getenv(
            'SWIFT_GKD_MICROBATCH_BACKWARD_TRACE', '0').lower() in {'1', 'true', 'yes'}
        self.output_dir = os.getenv('SWIFT_GKD_MICROBATCH_BACKWARD_DIR')
        self.tag = os.getenv('SWIFT_GKD_MICROBATCH_BACKWARD_TAG', '').strip()
        self.step = int(os.getenv('SWIFT_GKD_MICROBATCH_BACKWARD_STEP', '0'))
        self.handles = []
        self.records = {}
        self.seen_boundaries = set()
        self.expected_boundaries = set()

        if not self.enabled:
            return
        if not self.output_dir:
            raise ValueError(
                'SWIFT_GKD_MICROBATCH_BACKWARD_DIR is required when the trace is enabled.')
        if not self.tag or not re.fullmatch(r'[A-Za-z0-9_-]+', self.tag):
            raise ValueError(
                'SWIFT_GKD_MICROBATCH_BACKWARD_TAG must contain only letters, digits, _ and -.')
        if self.step < 0:
            raise ValueError('SWIFT_GKD_MICROBATCH_BACKWARD_STEP must be non-negative.')
        for argument in (
                'tensor_model_parallel_size',
                'pipeline_model_parallel_size',
                'context_parallel_size'):
            if int(getattr(trainer.args, argument, 1)) != 1:
                raise ValueError(
                    f'Micro-batch backward trace currently requires {argument}=1.')
        if trainer._is_debug_rank():
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(
                f'GKD native micro-batch backward trace enabled: step={self.step}, '
                f'tag={self.tag}, dir={self.output_dir}')

    def active(self, step):
        return self.enabled and int(step) == self.step and self.trainer._is_debug_rank()

    def prepare_train_step(self, step, debug_active):
        if not self.active(step):
            return
        if not debug_active:
            raise ValueError(
                'Micro-batch backward trace step must be inside the alignment debug window.')
        self.records = {}
        self.seen_boundaries = set()

    def _record(self, step, micro_batch):
        return self.records.setdefault((int(step), int(micro_batch)), {
            'format': 'swift_gkd_microbatch_backward_v2',
            'tag': self.tag,
            'step': int(step),
            'micro_batch': int(micro_batch),
            'provenance': None,
            'student_logits': None,
            'loss': None,
            'dlogits': None,
            'boundaries': {},
        })

    def capture_forward(self, data, labels, teacher_output, student_output, step, micro_batch):
        if not self.active(step):
            return
        record = self._record(step, micro_batch)
        teacher_logits = teacher_output.full_logits
        record['provenance'] = {
            'input_ids': self.trainer._tensor_identity(data.get('input_ids')),
            'position_ids': self.trainer._tensor_identity(data.get('position_ids')),
            'labels': self.trainer._tensor_identity(labels),
            'num_valid': int((labels != -100).sum().item()) if labels is not None else None,
            'student_logits': self.trainer._tensor_identity(student_output),
            'teacher_logits': self.trainer._tensor_identity(teacher_logits),
            'teacher_topk_logprobs': self.trainer._tensor_identity(
                teacher_output.topk_logprobs),
            'teacher_topk_indices': self.trainer._tensor_identity(
                teacher_output.topk_indices),
            'teacher_labels': self.trainer._tensor_identity(
                teacher_output.opsd_teacher_labels),
        }
        record['student_logits'] = student_output.detach().cpu().contiguous()
        if not student_output.requires_grad:
            raise ValueError('Micro-batch backward trace requires differentiable student logits.')

        def dlogits_hook(gradient):
            if record['dlogits'] is not None:
                raise RuntimeError(
                    f'dLogits was captured more than once for step={step}, micro_batch={micro_batch}.')
            record['dlogits'] = gradient.detach().cpu().contiguous()
            return gradient

        student_output.register_hook(dlogits_hook)

    def capture_loss(self, step, micro_batch, loss, metric, details):
        if not self.active(step):
            return
        record = self._record(step, micro_batch)
        if record['loss'] is not None:
            raise RuntimeError(
                f'Loss was captured more than once for step={step}, micro_batch={micro_batch}.')
        record['loss'] = {
            'backward_loss': float(loss.detach().float().cpu()),
            'metrics': {
                name: float(value.detach().float().cpu())
                for name, value in metric.items()
            },
            'details': dict(details or {}),
        }

    @staticmethod
    def _is_boundary(name):
        return (
            name in {'output_layer', 'decoder.final_layernorm'}
            or re.fullmatch(r'decoder\.layers\.\d+', name) is not None
        )

    def _backward_hook(self, model_index, name):

        def hook(module, grad_input, grad_output):
            context = self.trainer._backward_debug_context
            if context is None or not self.active(context['step']):
                return
            key = (context['step'], context['micro_batch'], model_index, name)
            if key in self.seen_boundaries:
                return
            self.seen_boundaries.add(key)
            input_gradient = self.trainer._first_tensor(grad_input)
            output_gradient = self.trainer._first_tensor(grad_output)
            # output_layer output_gradient is the already captured full dLogits.
            if name == 'output_layer':
                output_gradient = None
            record = self._record(context['step'], context['micro_batch'])
            record['boundaries'][f'model{model_index}.{name}'] = {
                'module_type': type(module).__name__,
                'input_gradient': (
                    input_gradient.detach().cpu().contiguous()
                    if input_gradient is not None else None),
                'output_gradient': (
                    output_gradient.detach().cpu().contiguous()
                    if output_gradient is not None else None),
            }

        return hook

    def register_hooks(self):
        matched = []
        for model_index, model in enumerate(self.trainer.unwrapped_models):
            for name, module in model.named_modules():
                if not self._is_boundary(name):
                    continue
                qualified_name = f'model{model_index}.{name}'
                handle = module.register_full_backward_hook(
                    self._backward_hook(model_index, name))
                self.handles.append(handle)
                self.expected_boundaries.add(qualified_name)
                matched.append(f'{qualified_name} ({type(module).__name__})')
        if not matched:
            raise ValueError('Micro-batch backward trace found no decoder boundaries.')
        logger.info(f'GKD native micro-batch backward hooks registered: {matched}')

    def _output_path(self, micro_batch):
        return os.path.join(
            self.output_dir,
            f'microbatch_backward_{self.tag}_step_{self.step:06d}_micro_{micro_batch:03d}.pt')

    def finalize_train_step(self, step, expected_microbatches):
        if not self.active(step):
            return
        expected_microbatches = int(expected_microbatches)
        expected_indices = set(range(expected_microbatches))
        actual_indices = {
            micro_batch for record_step, micro_batch in self.records
            if record_step == self.step
        }
        if actual_indices != expected_indices:
            raise RuntimeError(
                f'Micro-batch trace indices differ: expected={sorted(expected_indices)}, '
                f'actual={sorted(actual_indices)}.')
        for micro_batch in sorted(actual_indices):
            record = self.records[(self.step, micro_batch)]
            if (record['provenance'] is None
                    or record['student_logits'] is None
                    or record['loss'] is None
                    or record['dlogits'] is None):
                raise RuntimeError(
                    f'Micro-batch trace is incomplete at micro_batch={micro_batch}: '
                    f'provenance={record["provenance"] is not None}, '
                    f'student_logits={record["student_logits"] is not None}, '
                    f'loss={record["loss"] is not None}, '
                    f'dlogits={record["dlogits"] is not None}.')
            actual_boundaries = set(record['boundaries'])
            if actual_boundaries != self.expected_boundaries:
                raise RuntimeError(
                    f'Micro-batch {micro_batch} boundary set differs: '
                    f'missing={sorted(self.expected_boundaries - actual_boundaries)}, '
                    f'unexpected={sorted(actual_boundaries - self.expected_boundaries)}.')
            record['runtime'] = {
                'num_microbatches': expected_microbatches,
                'micro_batch_size': int(self.trainer.args.micro_batch_size),
                'global_batch_size': int(self.trainer.args.global_batch_size),
                'tensor_model_parallel_size': int(
                    getattr(self.trainer.args, 'tensor_model_parallel_size', 1)),
                'pipeline_model_parallel_size': int(
                    getattr(self.trainer.args, 'pipeline_model_parallel_size', 1)),
                'context_parallel_size': int(
                    getattr(self.trainer.args, 'context_parallel_size', 1)),
            }
            path = self._output_path(micro_batch)
            if os.path.exists(path):
                raise FileExistsError(f'Micro-batch backward trace already exists: {path}')
            torch.save(record, path)
            logger.info(
                f'GKD native micro-batch backward trace saved: '
                f'step={self.step}, micro_batch={micro_batch}, path={path}')
        self.records = {}
