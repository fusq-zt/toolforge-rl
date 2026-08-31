import pytest

from toolforge_rl.verifiers import normalize_numeric, normalize_qa, verify_answer, verify_math, verify_numeric, verify_qa


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("1,234", "1234"),
        ("$42 dollars", "42"),
        ("-3.50 kg", "-3.5"),
        ("2 / 4", "1/2"),
        (r"\boxed{7}", "7"),
    ],
)
def test_numeric_normalization(raw, normalized):
    assert normalize_numeric(raw) == normalized


def test_numeric_fraction_decimal_equivalence():
    assert verify_numeric("1/2", "0.5").correct


def test_numeric_tolerance():
    assert verify_numeric("0.3333333", "1/3").correct


def test_numeric_mismatch():
    assert not verify_numeric("4", "5").correct


def test_numeric_parse_failure():
    assert verify_numeric("none", "5").reason == "numeric_parse_failure"


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  New   York ", "new york"),
        ("An apple", "apple"),
        ("U.S.A.", "u s"),
    ],
)
def test_qa_normalization(raw, normalized):
    assert normalize_qa(raw) == normalized


def test_qa_exact():
    assert verify_qa("Paris", "paris").correct


def test_qa_alias():
    assert verify_qa("NYC", "New York City", aliases=["NYC"]).correct


def test_qa_not_substring_match():
    assert not verify_qa("York", "New York").correct


def test_registry_dispatch_numeric():
    assert verify_answer("gsm8k", "42", "42").correct


def test_registry_dispatch_qa():
    assert verify_answer("hotpotqa", "The Hague", "Hague").correct


def test_registry_unknown():
    with pytest.raises(ValueError):
        verify_answer("unknown", "a", "b")


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        (r"\frac{1}{2}", "0.5"),
        (r"\sqrt{4}", "2"),
        (r"x^2+2x+1", r"(x+1)^2"),
        (r"\boxed{7}", "7"),
    ],
)
def test_math_equivalence(prediction, reference):
    assert verify_math(prediction, reference).correct
