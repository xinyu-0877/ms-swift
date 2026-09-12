# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
import torch.distributed as dist
from dataclasses import asdict
from transformers.utils import is_torch_npu_available
from typing import List, Optional, Union

from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.trainers import MegatronEmbeddingTrainer, MegatronRerankerTrainer, MegatronTrainer
from swift.pipelines import SwiftSft
from swift.utils import append_to_jsonl, get_logger, is_last_rank, plot_images

if is_torch_npu_available():
    # Enable Megatron on Ascend NPU
    from mindspeed.megatron_adaptor import repatch

    from swift.model.npu_patcher import patch_mindspeed_te_cp_implementation
else:
    repatch = None
    patch_mindspeed_te_cp_implementation = None

logger = get_logger()


def _configure_strict_fp32(args) -> None:
    if os.getenv('SWIFT_GKD_STRICT_FP32', '0') != '1':
        return
    if args.torch_dtype != torch.float32 or args.fp16 or args.bf16:
        raise ValueError(
            'SWIFT_GKD_STRICT_FP32=1 requires --torch_dtype float32 '
            '--fp16 false --bf16 false.')

    # Keep FP32 matmuls from silently selecting TF32/HF32 accelerator modes.
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
    npu = getattr(torch, 'npu', None)
    if npu is not None:
        for backend_name in ('matmul', 'conv'):
            backend = getattr(npu, backend_name, None)
            if backend is not None and hasattr(backend, 'allow_hf32'):
                backend.allow_hf32 = False
    logger.info('Strict FP32 execution enabled; TF32/HF32 accelerator modes are disabled where available.')


class MegatronSft(SwiftSft):
    args_class = MegatronSftArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        if args.task_type == 'embedding':
            return MegatronEmbeddingTrainer(self.args, self.template)
        elif args.task_type in {'reranker', 'generative_reranker'}:
            return MegatronRerankerTrainer(self.args, self.template)
        else:
            return MegatronTrainer(self.args, self.template)

    def _set_seed(self):
        pass

    def __init__(self, args: Optional[Union[List[str], MegatronSftArguments]] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        _configure_strict_fp32(args)
        if repatch is not None:
            megatron_args = asdict(self.args)
            if args.attention_backend != 'local':
                # MindSpeed requires passing `use_flash_attn` to Megatron
                # to enable flash attention on Ascend NPU.
                args.use_flash_attn = True
                megatron_args['use_flash_attn'] = True
            patch_mindspeed_te_cp_implementation(megatron_args)
            repatch(megatron_args)
        template_cls = args.template_meta.template_cls
        if args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(**kwargs, download_model=args.mcore_model is None)
        self._prepare_template()
        args.save_args(args.output_dir)
        self.template.use_megatron = True

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        args.init_iters(train_dataset, val_dataset)
        trainer = self.prepare_trainer()
        try:
            trainer.train(train_dataset, val_dataset)
        finally:
            state = trainer.state
            self._handle_trainer_state(trainer, is_last_rank())
            self.train_msg.update({
                'last_model_checkpoint': state.last_model_checkpoint,
                'best_model_checkpoint': state.best_model_checkpoint,
                'best_metric': state.best_metric,
            })
            # Visualization
            if is_last_rank():
                images_dir = os.path.join(args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)

                jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
                append_to_jsonl(jsonl_path, self.train_msg, strict=False, write_on_rank='last')
        # Exceptions may cause the process to hang, preventing the exception from being propagated.
        # Therefore, destroy_process_group() should not be placed inside the finally block.
        if dist.is_initialized():
            dist.destroy_process_group()
        return self.train_msg


def megatron_sft_main(args: Optional[Union[List[str], MegatronSftArguments]] = None):
    return MegatronSft(args).main()
