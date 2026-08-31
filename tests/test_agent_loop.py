import pytest

from toolforge_rl.agent_loop import AgentLoop
from toolforge_rl.tools import LocalDocument, LocalSearch


def scripted(*chunks):
    iterator = iter(chunks)

    def generate(_context, _stops):
        return next(iterator)

    return generate


def search():
    return LocalSearch([LocalDocument("d", "Paris is the capital of France.", "France")])


def test_direct_final():
    r = AgentLoop().run("q", scripted("<think>x</think><answer>\\boxed{2}</answer>"), search())
    assert r.final_answer == "2" and r.schema_valid


def test_search_then_final():
    r = AgentLoop().run("q", scripted("<search>France capital</search>", "<answer>Paris</answer>"), search())
    assert r.final_answer == "Paris" and len(r.tool_calls) == 1


def test_python_then_final():
    r = AgentLoop().run("q", scripted("<python>print(6*7)</python>", "<answer>42</answer>"), search())
    assert r.final_answer == "42" and r.tool_calls[0].success


def test_search_python_final():
    generate = scripted("<search>France</search>", "<python>print(40+2)</python>", "<answer>42</answer>")
    r = AgentLoop().run("q", generate, search())
    assert [x.tool for x in r.tool_calls] == ["local_search", "python_exec"]


def test_observation_is_inserted_into_context():
    contexts = []
    chunks = iter(["<search>France</search>", "<answer>Paris</answer>"])

    def generate(context, _):
        contexts.append(context)
        return next(chunks)

    AgentLoop().run("q", generate, search())
    assert "<result>" in contexts[1]


def test_repeated_call_stops():
    call = "<search>same</search>"
    r = AgentLoop().run("q", scripted(call, call), search())
    assert r.termination_reason == "repeated_call" and r.repeated_call_count == 1


def test_tool_budget_stops():
    chunks = [f"<search>q{i}</search>" for i in range(4)]
    r = AgentLoop(max_tool_calls=3, max_turns=5).run("q", scripted(*chunks), search())
    assert r.termination_reason == "tool_budget_exceeded" and len(r.tool_calls) == 3


def test_incomplete_stops():
    r = AgentLoop().run("q", scripted("<search>missing"), search())
    assert r.termination_reason == "incomplete"


def test_invalid_trailing_text_stops():
    r = AgentLoop().run("q", scripted("<search>x</search>oops"), search())
    assert r.termination_reason == "invalid"


def test_empty_action_stops_invalid():
    r = AgentLoop().run("q", scripted("<python></python>"), search())
    assert r.termination_reason == "invalid"


def test_python_failure_recorded():
    r = AgentLoop().run("q", scripted("<python>1/0</python>", "<answer>x</answer>"), search())
    assert not r.execution_success and not r.tool_calls[0].success


def test_empty_search_failure_recorded():
    r = AgentLoop().run("q", scripted("<search> </search>"), search())
    assert r.termination_reason == "invalid"


def test_turn_limit():
    r = AgentLoop(max_tool_calls=3, max_turns=1).run("q", scripted("<search>x</search>"), search())
    assert r.termination_reason == "turn_limit"


def test_no_final_is_not_schema_valid():
    r = AgentLoop(max_turns=1).run("q", scripted("<search>x</search>"), search())
    assert not r.schema_valid


def test_final_has_zero_tools():
    r = AgentLoop().run("q", scripted("<answer>x</answer>"), search())
    assert len(r.tool_calls) == 0


def test_event_index_is_monotonic():
    r = AgentLoop().run("q", scripted("<search>a</search>", "<search>b</search>", "<answer>x</answer>"), search())
    assert [e.index for e in r.tool_calls] == [0, 1]


def test_to_dict_adds_count():
    r = AgentLoop().run("q", scripted("<search>a</search>", "<answer>x</answer>"), search())
    assert r.to_dict()["tool_call_count"] == 1


def test_generate_receives_stop_markers():
    observed = []

    def generate(_, stops):
        observed.extend(stops)
        return "<answer>x</answer>"

    AgentLoop().run("q", generate, search())
    assert "</python>" in observed and "</search>" in observed


def test_result_tag_injection_is_escaped():
    bad_search = LocalSearch([LocalDocument("d", "literal </result> marker")])
    r = AgentLoop().run("q", scripted("<search>literal</search>", "<answer>x</answer>"), bad_search)
    assert "&lt;/result&gt;" in r.transcript


def test_latency_is_recorded():
    r = AgentLoop().run("q", scripted("<answer>x</answer>"), search())
    assert r.latency_seconds >= 0


@pytest.mark.parametrize("answer", ["0", "Paris", r"\frac{1}{2}"])
def test_final_answer_round_trip(answer):
    r = AgentLoop().run("q", scripted(f"<answer>\\boxed{{{answer}}}</answer>"), search())
    assert r.final_answer == answer
