import pytest

from toolforge_rl.rewards import RewardInput, efficient_group_rewards, vanilla_reward


def item(correct=False, calls=0, schema=True, parsed=True, invalid=0, timeout=0, repeated=0):
    return RewardInput(correct, schema, parsed, calls, invalid, timeout, repeated)


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (item(True, 0), 1.05),
        (item(True, 1), 1.05),
        (item(True, 3), 1.05),
        (item(False, 0), 0.05),
        (item(False, 3), 0.05),
        (item(True, 1, schema=False), 1.00),
        (item(False, 0, schema=False), 0.00),
        (item(True, 1, parsed=False), 0.85),
        (item(False, 0, parsed=False), -0.15),
        (item(True, 1, invalid=1), 0.85),
        (item(True, 1, invalid=2), 0.65),
        (item(True, 1, invalid=3), 0.65),
        (item(False, 0, timeout=1), -0.15),
        (item(False, 0, repeated=1), -0.15),
        (item(True, 1, invalid=1, timeout=1, repeated=1), 0.65),
    ],
)
def test_vanilla_cases(sample, expected):
    assert vanilla_reward(sample).total == pytest.approx(expected)


def test_correct_zero_call_beats_wrong_zero_call():
    assert vanilla_reward(item(True, 0)).total > vanilla_reward(item(False, 0)).total


def test_execution_success_has_no_standalone_reward():
    assert vanilla_reward(item(False, 2)).answer == 0


def test_invalid_penalty_cap():
    assert vanilla_reward(item(True, 1, invalid=99)).invalid == -0.4


def test_schema_component():
    assert vanilla_reward(item(True)).format == 0.05


def test_parse_component():
    assert vanilla_reward(item(True, parsed=False)).parse == -0.2


def test_all_correct_same_calls_has_no_efficiency_penalty():
    out = efficient_group_rewards([item(True, 2) for _ in range(4)])
    assert all(x.efficiency == 0 for x in out)


def test_all_wrong_has_no_efficiency_penalty():
    out = efficient_group_rewards([item(False, c) for c in range(4)])
    assert all(x.efficiency == 0 and x.minimum_correct_calls is None for x in out)


def test_one_correct_sets_minimum():
    out = efficient_group_rewards([item(False, 0), item(True, 2), item(False, 3)])
    assert all(x.minimum_correct_calls == 2 for x in out)


def test_multiple_correct_different_calls():
    out = efficient_group_rewards([item(True, 1), item(True, 2), item(True, 3)])
    assert [x.efficiency for x in out] == pytest.approx([0, -0.15, -0.30])


def test_wrong_zero_call_gets_no_efficiency_advantage():
    out = efficient_group_rewards([item(False, 0), item(True, 2)])
    assert out[0].efficiency == 0
    assert out[1].efficiency == 0
    assert out[1].total > out[0].total


def test_wrong_multicall_not_efficiency_penalized():
    out = efficient_group_rewards([item(True, 0), item(False, 3)])
    assert out[1].efficiency == 0


def test_minimum_correct_is_zero_when_available():
    out = efficient_group_rewards([item(True, 0), item(True, 2)])
    assert out[1].minimum_correct_calls == 0
    assert out[1].efficiency == pytest.approx(-0.30)


@pytest.mark.parametrize(
    ("lam", "expected"),
    [(0.10, -0.20), (0.15, -0.30), (0.25, -0.50), (0.0, 0.0)],
)
def test_lambda_values(lam, expected):
    out = efficient_group_rewards([item(True, 0), item(True, 2)], efficiency_lambda=lam)
    assert out[1].efficiency == pytest.approx(expected)


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        efficient_group_rewards([item(True)], efficiency_lambda=-0.1)


def test_empty_group():
    assert efficient_group_rewards([]) == []


def test_correct_minimum_remains_highest_reward():
    out = efficient_group_rewards([item(True, 1), item(True, 3)])
    assert out[0].total > out[1].total


def test_incorrect_cannot_beat_clean_correct_due_to_cost():
    out = efficient_group_rewards([item(True, 3), item(False, 0)], efficiency_lambda=0.25)
    assert out[0].total > out[1].total


def test_timeout_and_invalid_share_cap():
    assert vanilla_reward(item(True, invalid=1, timeout=2)).invalid == -0.4


def test_repeated_and_invalid_share_cap():
    assert vanilla_reward(item(True, invalid=1, repeated=2)).invalid == -0.4


def test_parse_failure_and_invalid_are_additive():
    assert vanilla_reward(item(False, parsed=False, invalid=1)).total == pytest.approx(-0.35)


def test_format_reward_does_not_require_correctness():
    assert vanilla_reward(item(False, schema=True)).format == 0.05


def test_efficiency_preserves_base_components():
    base = vanilla_reward(item(True, 2))
    out = efficient_group_rewards([item(True, 1), item(True, 2)])[1]
    assert (out.answer, out.format, out.invalid, out.parse) == (
        base.answer, base.format, base.invalid, base.parse
    )


def test_tool_count_below_minimum_is_clamped_defensively():
    # Minimum is computed from the same group, so no normal item can be below it;
    # this assertion documents that the selected minimum itself is never rewarded.
    out = efficient_group_rewards([item(True, 1), item(True, 4)])
    assert out[0].efficiency == 0


def test_four_rollout_reference_group():
    out = efficient_group_rewards([item(True, 1), item(True, 2), item(False, 0), item(True, 3)])
    assert [round(x.total, 2) for x in out] == [1.05, 0.90, 0.05, 0.75]
