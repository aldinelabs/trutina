import mlx.core as mx

# MLX JIT-compiles fused CPU kernels with the host C++ toolchain; on machines whose
# Command Line Tools cannot link against the installed SDK this aborts the process.
# Tests run small models where compile buys nothing, so turn it off for the session.
mx.disable_compile()
