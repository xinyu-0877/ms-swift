#!/usr/bin/env python3
"""Fail fast when the SFT tracing Python integration was not deployed."""

import inspect
import os


def main():
    import swift.megatron.trainers.base as base_module
    from swift.megatron.trainers.base import BaseMegatronTrainer
    from swift.megatron.trainers.trainer import MegatronTrainer

    base_source = inspect.getsource(BaseMegatronTrainer)
    trainer_source = inspect.getsource(MegatronTrainer)
    checks = {
        'base_imports_sft_trace': 'SFTAttentionForwardTrace' in inspect.getsource(base_module),
        'base_prepares_trace': '_sft_attention_forward_trace.prepare_train_step' in base_source,
        'trainer_captures_provenance': 'trace.capture_provenance' in trainer_source,
        'trace_env_enabled': os.getenv('SWIFT_SFT_ATTENTION_FORWARD_TRACE') == '1',
    }
    print(checks)
    missing = [name for name, value in checks.items() if not value]
    if missing:
        raise RuntimeError(
            'SFT Layer-0 A-G integration is incomplete: ' + ', '.join(missing) +
            '. Sync swift/megatron/trainers/base.py, trainer.py, and '
            'sft_attention_forward_debug.py from the same checkout.')


if __name__ == '__main__':
    main()
