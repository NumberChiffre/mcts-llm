def parse_integer_answer(answer: str, only_first_line: bool = True) -> int:
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)
    except (ValueError, IndexError):
        answer = 0
    return answer
