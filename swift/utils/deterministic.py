# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional deterministic execution settings for GPU reproducibility checks.

The settings in this module are deliberately opt-in.  Enable them with
``SWIFT_GKD_DETERMINISTIC=1`` when comparing two repeated GPU runs.  They do
not change tensor dtypes or the GPU/NPU implementation path.
"""

import os
import random
from typing import Optional


def set_swift_deterministic(seed: Optional[int] = None) -> bool:
    """Enable deterministic PyTorch/CUDA behavior when explicitly requested.

    Returns ``True`` when the mode was enabled and ``False`` otherwise.  The
    cuBLAS workspace variable is set as a best effort; it should preferably be
    exported by the shell *before* Python imports torch.
    """
    if os.getenv('SWIFT_GKD_DETERMINISTIC', '0').lower() not in {'1', 'true', 'yes', 'on'}:
        return False

    if seed is None:
        seed = int(os.getenv('SWIFT_GKD_DETERMINISTIC_SEED', '42'))

    # Set this before CUDA kernels are initialized when possible.  A shell
    # export remains the reliable method for a process that already imported
    # torch (which is common for the Swift CLI).
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Disable TF32 matrix multiply/convolution paths for reproducibility.
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False

    # warn_only keeps training usable when a fused op has no deterministic
    # implementation; the warning identifies that limitation in the log.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError) as exc:
        print(f'[swift] deterministic algorithms unavailable: {exc}')

    print(f'[swift] deterministic mode enabled (seed={seed})')
    return True
