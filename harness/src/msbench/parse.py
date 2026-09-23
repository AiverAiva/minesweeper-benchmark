import re

ACTION_LINE = re.compile(
    r"^ACTION: (open|flag|chord) row=([1-9][0-9]*) col=([1-9][0-9]*)$"
)


def parse_action(text: str) -> dict | None:
    """Return the single ACTION line, or None when the reply is not exactly one."""
    found = []
    for line in text.splitlines():
        match = ACTION_LINE.match(line.strip())
        if match is None:
            continue
        found.append(
            {
                "kind": match.group(1),
                "at": {"row": int(match.group(2)), "col": int(match.group(3))},
            }
        )
    if len(found) != 1:
        return None
    return found[0]


def format_action(action: dict) -> str:
    at = action["at"]
    return f"ACTION: {action['kind']} row={at['row']} col={at['col']}"
