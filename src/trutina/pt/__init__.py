"""PyTorch port of the training path, for CUDA machines (e.g. Colab).

Inference and latency measurement stay on MLX. Module and parameter names mirror the MLX
classes exactly, so `export_trainable` writes a safetensors file that
`trutina.train.load_trainable` loads unchanged.
"""
