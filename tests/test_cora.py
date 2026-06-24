from src.data.cora import load_cora


def test_load_cora_structure():
    data = load_cora()
    assert data.num_nodes == 2708
    assert data.x.shape[1] == 1433
    assert int(data.y.max()) == 6  # 7 classes (0..6)
    for mask in ("train_mask", "val_mask", "test_mask"):
        assert hasattr(data, mask)
        assert data[mask].sum() > 0
