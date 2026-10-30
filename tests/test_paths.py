"""Unit tests for the path parsing (run with ``python -m pytest tests``)."""
from src.data.paths import resolve_path


def test_forward_only():
    assert resolve_path("A;B;C") == [("A", "B", False), ("B", "C", False)]


def test_back_click_returns_to_previous_page():
    # A -> B -> C, back to B, then D is clicked *from B*
    assert resolve_path("A;B;C;<;D") == [
        ("A", "B", False), ("B", "C", False), ("C", "B", True), ("B", "D", False)]


def test_consecutive_back_clicks_unwind_the_stack():
    # two back clicks from C land on A, so D is clicked from A
    moves = resolve_path("A;B;C;<;<;D")
    assert moves[-1] == ("A", "D", False)
    assert [m[2] for m in moves] == [False, False, True, True, False]


def test_impossible_back_click_is_ignored():
    assert resolve_path("A;<;B") == []
