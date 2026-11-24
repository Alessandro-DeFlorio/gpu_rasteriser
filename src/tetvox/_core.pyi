"""tetvox C++/CUDA core"""



cuda_enabled: bool = True

def device_count() -> int:
    """Number of visible CUDA devices (always 0 in a CPU-only build)."""
