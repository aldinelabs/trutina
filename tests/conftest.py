try:
    import mlx.core as mx
except ImportError:  # board tests do not depend on Apple-only MLX
    mx = None

# MLX JIT-compiles fused CPU kernels with the host C++ toolchain; on machines whose
# Command Line Tools cannot link against the installed SDK this aborts the process.
# Tests run small models where compile buys nothing, so turn it off for the session.
if mx is not None:
    mx.disable_compile()
