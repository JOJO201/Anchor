from anchoradd import stateless_anchor_index, stateless_apply_decision


def test_stateless_sampling_is_reproducible() -> None:
    first = [stateless_anchor_index(17, 5, 2, step) for step in range(20)]
    second = [stateless_anchor_index(17, 5, 2, step) for step in range(20)]
    assert first == second
    assert all(0 <= value < 17 for value in first)
    assert stateless_apply_decision(0.0, 1, 0, 0) is False
    assert stateless_apply_decision(1.0, 1, 0, 0) is True
