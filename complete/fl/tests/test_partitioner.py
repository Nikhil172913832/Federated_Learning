from fl.partitioning import build_partitioner


def test_build_partitioner_defaults_to_iid():
    part = build_partitioner(num_partitions=3, cfg=None)
    assert part is not None


def test_build_partitioner_label_skew():
    cfg = {"type": "label_skew", "params": {"alpha": 0.3}}
    part = build_partitioner(num_partitions=3, cfg=cfg)
    assert part is not None


def test_build_partitioner_quantity_skew():
    cfg = {"type": "quantity_skew", "params": {"min_size": 10}}
    part = build_partitioner(num_partitions=3, cfg=cfg)
    assert part is not None


