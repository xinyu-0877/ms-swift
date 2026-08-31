#!/usr/bin/env python3
"""Check if fp32_residual_connection is supported in current environment."""
import sys

def check_gpu_support():
    """Check GPU/NVIDIA environment."""
    print("=== GPU/NVIDIA Environment ===")
    try:
        from mcore_bridge import ModelConfig
        import inspect
        sig = inspect.signature(ModelConfig.__init__)
        params = list(sig.parameters.keys())

        if 'fp32_residual_connection' in params:
            print("✓ fp32_residual_connection parameter exists in ModelConfig")
            # Try to get default value
            param = sig.parameters['fp32_residual_connection']
            print(f"  Default value: {param.default}")
            return True
        else:
            print("✗ fp32_residual_connection NOT found in ModelConfig parameters")
            print(f"  Available parameters ({len(params)}): {params[:10]}...")
            return False
    except ImportError as e:
        print(f"✗ Cannot import mcore_bridge: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking ModelConfig: {e}")
        return False

def check_npu_support():
    """Check NPU/Ascend environment."""
    print("\n=== NPU/Ascend Environment ===")
    try:
        # NPU requires importing swift.megatron first to patch mcore_bridge
        import swift.megatron
        print("✓ swift.megatron imported (MindSpeed patches applied)")

        from mcore_bridge import ModelConfig
        import inspect
        sig = inspect.signature(ModelConfig.__init__)
        params = list(sig.parameters.keys())

        if 'fp32_residual_connection' in params:
            print("✓ fp32_residual_connection parameter exists in ModelConfig")
            param = sig.parameters['fp32_residual_connection']
            print(f"  Default value: {param.default}")
            return True
        else:
            print("✗ fp32_residual_connection NOT found in ModelConfig parameters")
            print(f"  Available parameters ({len(params)}): {params[:10]}...")

            # Check if it's in the config class itself
            if hasattr(ModelConfig, 'fp32_residual_connection'):
                print("  Note: Found as class attribute")
            return False
    except ImportError as e:
        print(f"✗ Cannot import required modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking ModelConfig: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_transformer_engine():
    """Check if TransformerEngine is available and its config."""
    print("\n=== TransformerEngine Check ===")
    try:
        import transformer_engine
        print(f"✓ TransformerEngine version: {transformer_engine.__version__}")

        from transformer_engine.common import recipe
        print("✓ TransformerEngine recipe available")

        # Check TransformerConfig
        try:
            from megatron.core.transformer.transformer_config import TransformerConfig
            import inspect
            sig = inspect.signature(TransformerConfig.__init__)
            params = list(sig.parameters.keys())

            if 'fp32_residual_connection' in params:
                print("✓ fp32_residual_connection exists in TransformerConfig")
                return True
            else:
                print("✗ fp32_residual_connection NOT in TransformerConfig")
                print(f"  Available: {params[:15]}...")
                return False
        except ImportError:
            print("  TransformerConfig not available")
            return False

    except ImportError:
        print("✗ TransformerEngine not installed (GPU-only package)")
        return False

def main():
    print("Checking fp32_residual_connection support...\n")

    # Detect environment
    try:
        import torch
        if hasattr(torch, 'npu') and torch.npu.is_available():
            print("Detected: NPU environment")
            supported = check_npu_support()
        elif torch.cuda.is_available():
            print("Detected: GPU environment")
            supported = check_gpu_support()
            check_transformer_engine()
        else:
            print("Detected: CPU-only environment")
            supported = check_gpu_support()
    except Exception as e:
        print(f"Error detecting environment: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    if supported:
        print("RESULT: fp32_residual_connection IS supported")
        print("You can use: --megatron_extra_kwargs '{\"fp32_residual_connection\": true}'")
        sys.exit(0)
    else:
        print("RESULT: fp32_residual_connection NOT supported")
        print("\nPossible reasons:")
        print("1. Megatron-Core version too old")
        print("2. MindSpeed version doesn't include this feature")
        print("3. mcore_bridge doesn't expose this parameter")
        print("\nWorkaround: Check if your framework supports it via alternative config paths")
        sys.exit(1)

if __name__ == '__main__':
    main()
