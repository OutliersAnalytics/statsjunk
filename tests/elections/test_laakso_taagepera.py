import pytest

from statsjunk.elections import FragmentationResult, compute_laakso_taagepera


def test_single_winner_is_one():
    result = compute_laakso_taagepera([100, 0, 0])

    assert isinstance(result, FragmentationResult)
    assert result.effective_number == pytest.approx(1.0)
    assert result.largest_share == pytest.approx(1.0)
    assert result.n_options == 1


def test_two_way_tie_is_two():
    result = compute_laakso_taagepera([50, 50])

    assert result.effective_number == pytest.approx(2.0)
    assert result.largest_share == pytest.approx(0.5)
    assert result.n_options == 2


def test_k_way_tie_is_k():
    result = compute_laakso_taagepera([10] * 7)

    assert result.effective_number == pytest.approx(7.0)
    assert result.n_options == 7


def test_dominant_option_pulls_effective_number_down():
    result = compute_laakso_taagepera([80, 10, 10])

    assert 1 < result.effective_number < 2
    assert result.largest_share == pytest.approx(0.8)


def test_zero_vote_options_are_ignored():
    a = compute_laakso_taagepera([60, 40])
    b = compute_laakso_taagepera([60, 40, 0, 0])

    assert a.effective_number == pytest.approx(b.effective_number)
    assert b.n_options == 2


def test_raises_for_empty():
    with pytest.raises(ValueError, match="At least one option"):
        compute_laakso_taagepera([])


def test_raises_for_negative():
    with pytest.raises(ValueError, match="non-negative"):
        compute_laakso_taagepera([10, -1])


def test_raises_for_zero_total():
    with pytest.raises(ValueError, match="must be positive"):
        compute_laakso_taagepera([0, 0, 0])
