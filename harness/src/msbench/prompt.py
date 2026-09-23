import hashlib

# Frozen model-facing prefix. The board text is appended per turn.
# PROMPT_SHA256 covers this prefix only, so a board change does not look like a prompt change.
PREFIX = """\
Reply with exactly one line:
ACTION: open|flag|chord row=<row> col=<col>

Coordinates are positive integers. Row 1 is the top row. Column 1 is the left column.
# hidden, F flag, . opened zero, 1-8 opened counts, * exploded mine.
Any other text is ignored.

"""

PROMPT_SHA256 = hashlib.sha256(PREFIX.encode("utf-8")).hexdigest()


def render_prompt(board_text: str) -> str:
    return PREFIX + board_text
