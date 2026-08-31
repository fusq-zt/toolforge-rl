from toolforge_rl.protocols.toolstar import (
    ActionKind,
    extract_final_answer,
    parse_next_action,
    render_result,
    validate_transcript,
)


def test_parses_search_action():
    action = parse_next_action("<think>x</think><search>alpha beta</search>")
    assert action.kind == ActionKind.SEARCH
    assert action.content == "alpha beta"


def test_parses_python_action_and_maps_tool_name():
    action = parse_next_action("<python>print(6 * 7)</python>\n")
    assert action.kind == ActionKind.PYTHON
    assert action.content == "print(6 * 7)"


def test_parses_final_action():
    action = parse_next_action("<answer>\\boxed{42}</answer>")
    assert action.kind == ActionKind.FINAL


def test_incomplete_action_is_not_executed():
    assert parse_next_action("<search>alpha").kind == ActionKind.INCOMPLETE


def test_plain_text_is_incomplete():
    assert parse_next_action("still reasoning").kind == ActionKind.INCOMPLETE


def test_rejects_trailing_text():
    action = parse_next_action("<search>alpha</search> trailing")
    assert action.kind == ActionKind.INVALID


def test_rejects_empty_body():
    assert parse_next_action("<python>  </python>").kind == ActionKind.INVALID


def test_uses_last_complete_action():
    action = parse_next_action("<search>a</search><result>x</result><search>b</search>")
    assert action.kind == ActionKind.SEARCH
    assert action.content == "b"


def test_result_rendering():
    assert render_result("ok") == "\n<result>\nok\n</result>\n"


def test_result_prevents_tag_injection():
    assert "</result>bad" not in render_result("</result>bad")


def test_result_is_bounded():
    assert len(render_result("x" * 500, max_chars=100)) < 140


def test_extracts_simple_box():
    assert extract_final_answer("<answer>\\boxed{7}</answer>") == "7"


def test_extracts_nested_box():
    text = r"<answer>The result is \boxed{\frac{1}{2}}</answer>"
    assert extract_final_answer(text) == r"\frac{1}{2}"


def test_extracts_last_answer_and_last_box():
    text = r"<answer>\boxed{bad}</answer><answer>x \boxed{good}</answer>"
    assert extract_final_answer(text) == "good"


def test_answer_without_box_falls_back_to_body():
    assert extract_final_answer("<answer>Paris</answer>") == "Paris"


def test_missing_answer():
    assert extract_final_answer("<think>x</think>") is None


def test_valid_direct_transcript():
    report = validate_transcript("<think>x</think><answer>\\boxed{2}</answer>")
    assert report.valid
    assert report.tool_calls == 0


def test_valid_tool_transcript():
    text = (
        "<think>x</think><python>print(2)</python><result>2</result>"
        "<think>done</think><answer>\\boxed{2}</answer>"
    )
    assert validate_transcript(text).valid


def test_rejects_orphan_result():
    report = validate_transcript("<result>x</result><answer>\\boxed{x}</answer>")
    assert not report.valid
    assert "orphan result tag" in report.errors


def test_rejects_missing_result():
    report = validate_transcript("<search>x</search><answer>\\boxed{x}</answer>")
    assert not report.valid


def test_rejects_budget_overrun():
    steps = "".join(f"<search>{i}</search><result>{i}</result>" for i in range(4))
    report = validate_transcript(steps + "<answer>\\boxed{x}</answer>")
    assert not report.valid
    assert any("budget exceeded" in e for e in report.errors)


def test_allows_partial_when_requested():
    assert validate_transcript("<think>x</think>", require_final=False).valid


def test_rejects_unbalanced_tag():
    report = validate_transcript("<think>x<answer>\\boxed{x}</answer>")
    assert not report.valid
    assert any("unbalanced think" in e for e in report.errors)
