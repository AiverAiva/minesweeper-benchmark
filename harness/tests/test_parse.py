import hashlib

from msbench.parse import format_action, parse_action
from msbench.prompt import PREFIX, PROMPT_SHA256, render_prompt

FROZEN_PREFIX = """\
Reply with exactly one line:
ACTION: open|flag|chord row=<row> col=<col>

Coordinates are positive integers. Row 1 is the top row. Column 1 is the left column.
# hidden, F flag, . opened zero, 1-8 opened counts, * exploded mine.
Any other text is ignored.

"""


def test_prefix_is_frozen():
    assert PREFIX == FROZEN_PREFIX
    assert PROMPT_SHA256 == hashlib.sha256(FROZEN_PREFIX.encode("utf-8")).hexdigest()


def test_prompt_appends_the_board():
    board = "status: playing\n"
    assert render_prompt(board) == PREFIX + board


def test_one_action_line_ignores_surrounding_prose():
    action = parse_action("Sure.\nACTION: open row=3 col=1\nThanks.")
    assert action == {"kind": "open", "at": {"row": 3, "col": 1}}


def test_two_action_lines_are_rejected():
    assert parse_action("ACTION: open row=1 col=1\nACTION: flag row=2 col=2") is None


def test_row_zero_is_rejected():
    assert parse_action("ACTION: open row=0 col=1") is None


def test_format_action_roundtrip():
    action = {"kind": "chord", "at": {"row": 12, "col": 3}}
    assert parse_action("note\n" + format_action(action) + " \n") == action
