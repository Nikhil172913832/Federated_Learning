from fl.task import apply_transforms


def test_apply_transforms_structure():
    batch = {"image": [0, 1], "label": [0, 1]}
    out = apply_transforms(batch)
    assert "image" in out and "label" in out
    assert len(out["image"]) == 2


