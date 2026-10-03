"""The S8 frontier (analysis/frontier.py): exhaustive plurality voting, ties, abstentions, costs and interpolation, on
hand-built cases whose answers are worked out by hand."""

from math import comb

import numpy as np
import pytest

from ape.analysis import frontier as fr

A, B, C, NONE = 0, 1, 2, fr.ABSTAIN


def test_plurality_over_every_subset_with_ties_counted_as_expected_success():
    # Runs: A (right), B (wrong), A (right).
    # k=1: 2/3. k=2: {A,B} tie 0.5, {A,A} 1, {B,A} tie 0.5 -> 2/3. k=3: A wins 2-1 -> 1.
    ts, bad = fr.s8_success(np.array([[A, B, A]]), np.array([[1.0, 0.0, 1.0]]))
    assert np.allclose(ts, [[2 / 3, 2 / 3, 1.0]]) and bad == 0


def test_abstentions_cast_no_vote_and_an_all_abstaining_subset_fails():
    # Runs: A (right), no answer, B (wrong).
    # k=1: 1/3. k=2: {A,-} 1, {A,B} 0.5, {-,B} 0 -> 0.5. k=3: A vs B tie -> 0.5.
    ts, _ = fr.s8_success(np.array([[A, NONE, B]]), np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(ts, [[1 / 3, 0.5, 0.5]])
    ts, _ = fr.s8_success(np.array([[NONE, NONE]]), np.array([[0.0, 0.0]]))
    assert np.allclose(ts, 0.0)


def test_a_three_way_tie_scores_its_share_and_wrong_majorities_lose():
    # Runs: A (right), B, C (both wrong, distinct): k=3 three-way tie -> 1/3.
    ts, _ = fr.s8_success(np.array([[A, B, C]]), np.array([[1.0, 0.0, 0.0]]))
    assert ts[0, 2] == pytest.approx(1 / 3)
    # Runs: A (right), B, B (a shared wrong answer): k=3 -> B wins -> 0.
    ts, _ = fr.s8_success(np.array([[A, B, B]]), np.array([[1.0, 0.0, 0.0]]))
    assert ts[0, 2] == 0.0 and ts[0, 1] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_runs_of_one_key_that_disagree_on_success_are_counted():
    _, bad = fr.s8_success(np.array([[A, A, B], [A, B, C]]), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert bad == 1


def test_k1_is_the_mean_run_success_and_the_estimator_is_unbiased_for_a_fresh_ensemble():
    """With distinct wrong answers and no abstentions, a 3-run vote succeeds with p³ + 3p²(1−p) + p(1−p)²
    (one right answer in a three-way tie wins a third of the time)."""
    rng = np.random.default_rng(0)
    p = 0.6
    ok = rng.random((20_000, 8)) < p
    codes = np.where(ok, 0, 1 + np.arange(8)[None, :])
    ts, _ = fr.s8_success(codes, ok.astype(float))
    assert ts[:, 0].mean() == pytest.approx(ok.mean())
    assert ts[:, 2].mean() == pytest.approx(p**3 + 3 * p**2 * (1 - p) + p * (1 - p) ** 2, abs=0.006)


def test_key_codes_map_answers_per_task_by_first_occurrence():
    assert fr.key_codes([["x", "y", "x", None], ["z", None, "z", "z"]]).tolist() == [[0, 1, 0, -1], [0, -1, 0, 0]]


def test_cost_of_s8_k_is_k_runs_summed_or_the_expected_maximum_for_parallel_members():
    cost = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(fr.s8_cost(cost, "sum"), [[2.0, 4.0, 6.0]])
    # max over k-subsets: k=1 mean 2; k=2 {1,2}->2, {1,3}->3, {2,3}->3 -> 8/3; k=3 -> 3.
    assert np.allclose(fr.s8_cost(cost, "max"), [[2.0, 8 / 3, 3.0]])
    with pytest.raises(ValueError):
        fr.s8_cost(cost, "mean")


def _frontier() -> fr.Frontier:
    """Two tasks, three runs, run cost 1 everywhere: S8 costs 1, 2, 3."""
    codes = np.array([[A, B, A], [A, B, C]])
    success = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return fr.build(codes, success, np.ones((2, 3)))


def test_interpolation_is_linear_in_cost_between_adjacent_k_per_task():
    f = _frontier()
    # task curves: [2/3, 2/3, 1] and [1/3, 1/3, 1/3]
    assert np.allclose(f.cost, [1, 2, 3]) and np.allclose(f.success, [0.5, 0.5, 2 / 3])
    vals, info = fr.interpolate(f, 2.5)
    assert info["status"] == "inside" and info["k"] == 2 and info["lambda"] == pytest.approx(0.5)
    assert np.allclose(vals, [0.5 * 2 / 3 + 0.5 * 1.0, 1 / 3])
    vals, info = fr.interpolate(f, 2.0)  # exactly on a point: S8(2)
    assert info["k"] == 2 and info["lambda"] == 0.0 and np.allclose(vals, [2 / 3, 1 / 3])
    vals, info = fr.interpolate(f, 3.0)  # the last point is inside
    assert info["status"] == "inside" and info["k"] == 3 and np.allclose(vals, [1.0, 1 / 3])


def test_beyond_the_frontier_is_not_extrapolated_and_below_meets_s8_1_flagged():
    f = _frontier()
    vals, info = fr.interpolate(f, 3.01)
    assert vals is None and info["status"] == "beyond" and info["max_cost"] == 3.0
    vals, info = fr.interpolate(f, 0.4)
    assert info["status"] == "below" and np.allclose(vals, [2 / 3, 1 / 3])


def test_an_unmetered_meter_gives_a_degenerate_frontier():
    f = fr.build(np.array([[A, B]]), np.array([[1.0, 0.0]]), np.zeros((1, 2)))
    assert fr.interpolate(f, 0.0)[1]["status"] == "degenerate"
    f = fr.build(np.array([[A, B]]), np.array([[1.0, 0.0]]), np.full((1, 2), np.nan))
    assert fr.interpolate(f, 1.0)[0] is None


def test_subset_masks_cover_every_subset_once():
    masks, sizes = fr.subset_masks(8)
    assert len(masks) == 255 and all((sizes == k).sum() == comb(8, k) for k in range(1, 9))
    assert len({tuple(m) for m in masks}) == 255
