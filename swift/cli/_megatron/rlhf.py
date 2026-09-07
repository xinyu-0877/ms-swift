# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys


def _use_ray() -> bool:
    if '--use_ray' not in sys.argv:
        return False
    idx = sys.argv.index('--use_ray')
    sys.argv.pop(idx)
    if idx < len(sys.argv) and sys.argv[idx].lower() in ('true', 'false'):
        val = sys.argv.pop(idx).lower() == 'true'
        return val
    return True


if __name__ == '__main__':
    if _use_ray():
        from swift.ray.megatron.pipeline import main as ray_main
        ray_main()
    else:
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        # These variables must be present before importing torch/CUDA.  The
        # full seed and PyTorch deterministic-algorithm setup is performed by
        # swift.utils.deterministic after the Megatron entry point starts.
        if os.getenv('SWIFT_GKD_DETERMINISTIC', '0').lower() in ('1', 'true', 'yes', 'on'):
            det_seed = os.getenv('SWIFT_GKD_DETERMINISTIC_SEED', '42')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')
            os.environ.setdefault('PYTHONHASHSEED', det_seed)
        from swift.megatron import megatron_rlhf_main
        megatron_rlhf_main()
