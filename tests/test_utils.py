import pytest

from mcts_llm.utils import parse_integer_answer


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("The answer is 42.", 42),
        ("The result is 123\nAdditional information", 123),
        ("No numbers here", 0),
        ("The answer is 42.5", 42),
        ("Multiple numbers: 10, 20, 30", 30),
    ],
)
def test_parse_integer_answer(input_str, expected_output):
    assert parse_integer_answer(input_str) == expected_output


def test_parse_integer_answer_multiline():
    input_str = "The answer is 42\nBut wait, there's more: 100"
    assert parse_integer_answer(input_str) == 42
    assert parse_integer_answer(input_str, only_first_line=False) == 100
