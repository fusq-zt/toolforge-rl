import pytest

from toolforge_rl.tools.python_exec import execute_python


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("print(1 + 1)", "2"),
        ("print(6 * 7)", "42"),
        ("print(2 ** 10)", "1024"),
        ("print(sum(range(5)))", "10"),
        ("print(round(1 / 3, 3))", "0.333"),
        ("print(min([3, 1, 2]))", "1"),
        ("print(max([3, 1, 2]))", "3"),
        ("print(sorted([3, 1, 2]))", "[1, 2, 3]"),
        ("print(list(map(abs, [-2, 3])))", "[2, 3]"),
        ("print(len({1, 2, 2}))", "2"),
        ("import math\nprint(math.factorial(6))", "720"),
        ("import statistics\nprint(statistics.mean([1,2,3]))", "2"),
        ("from fractions import Fraction\nprint(Fraction(2, 4))", "1/2"),
        ("from decimal import Decimal\nprint(Decimal('0.1') + Decimal('0.2'))", "0.3"),
        ("from itertools import product\nprint(len(list(product(range(2), repeat=3))))", "8"),
        ("from collections import Counter\nprint(Counter('aba')['a'])", "2"),
        ("x=[i*i for i in range(4)]\nprint(x)", "[0, 1, 4, 9]"),
        ("print(pow(3, 20, 5))", "1"),
    ],
)
def test_python_success_cases(code, expected):
    result = execute_python(code)
    assert result.ok
    assert result.output == expected


@pytest.mark.parametrize(
    "code",
    [
        "import os\nprint(os.getcwd())",
        "import sys\nprint(sys.version)",
        "import socket",
        "import subprocess",
        "open('/tmp/forbidden', 'w')",
        "eval('1+1')",
        "exec('print(1)')",
        "compile('1', 'x', 'eval')",
        "input()",
        "globals()",
        "().__class__",
        "global x\nx=1",
    ],
)
def test_python_blocks_unsafe_constructs(code):
    result = execute_python(code)
    assert not result.ok
    assert "PYTHON_ERROR" in result.output


def test_python_empty_program():
    result = execute_python("   ")
    assert not result.ok and result.error_type == "EmptyProgram"


def test_python_runtime_error_is_structured():
    result = execute_python("print(1/0)")
    assert not result.ok and result.error_type == "ZeroDivisionError"


def test_python_syntax_error_is_structured():
    result = execute_python("if:")
    assert not result.ok and result.error_type == "SyntaxError"


def test_python_no_stdout_is_explicit():
    result = execute_python("x = 3")
    assert result.ok and result.output == "<no stdout>"


def test_python_output_is_bounded():
    result = execute_python("print('x' * 1000)", max_output_chars=100)
    assert result.ok and len(result.output) <= 100


def test_python_timeout():
    result = execute_python("while True:\n pass", timeout_seconds=0.2)
    assert not result.ok
    assert result.error_type == "TimeoutExpired"
