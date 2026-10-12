from trutina.readout import FORMAT_VERSION, answer_prefix, prompt
from trutina.schema import Question


def test_sandwich_letter_prompt():
    q = Question("q", "Which fruit is yellow?", "choice", ("apple", "banana"), ("", "a long fruit"))
    p = prompt("A fruit bowl.", q, "sandwich", "letter")
    assert p.count("(A) apple") == 2 and p.count("(B) banana: a long fruit") == 2
    assert p.index("Input:\nA fruit bowl.") > p.index("(B) banana")  # schema, input, schema
    assert answer_prefix("letter") == "Best answer: ("
    assert FORMAT_VERSION == 1
