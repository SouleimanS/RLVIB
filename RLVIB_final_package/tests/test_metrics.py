"""Smoke tests for the torch-free eval metrics (run in web sessions; no GPU stack needed)."""
from rlvib.eval.metrics import accuracy, parse_choice, parse_yes_no


def test_parse_yes_no():
    assert parse_yes_no("Yes, there is.") == "yes"
    assert parse_yes_no("No.") == "no"
    assert parse_yes_no("correct") == "yes"               # first-word synonym
    assert parse_yes_no("I think yes actually") == "yes"  # fallback finds a literal yes/no
    assert parse_yes_no("") is None
    assert parse_yes_no("hmm") is None


def test_parse_choice():
    assert parse_choice("(B)") == "B"
    assert parse_choice("B) the dog barking") == "B"
    assert parse_choice("none of these") is None


def test_accuracy():
    r = accuracy(["yes", "no", None], ["yes", "yes", "no"])
    assert r["n"] == 3
    assert r["correct"] == 1
    assert abs(r["accuracy"] - 1 / 3) < 1e-9
    assert abs(r["parse_rate"] - 2 / 3) < 1e-9
