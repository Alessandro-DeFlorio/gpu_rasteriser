import tetvox


def test_device_count_matches_build() -> None:
    # The dev machine has a GPU; a CPU-only build must report none (ADR 0008).
    if tetvox.cuda_enabled:
        assert tetvox.device_count() >= 1
    else:
        assert tetvox.device_count() == 0
