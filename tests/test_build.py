import tetvox


def test_cuda_device_visible() -> None:
    assert tetvox.device_count() >= 1
