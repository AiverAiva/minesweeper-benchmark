"""The 3×3 textbook board: mines at (1, 1) and (3, 3), first click (1, 3)."""

OPENING = (
    "status: playing\n"
    "mines_total: 2\n"
    "flags_left: 2\n"
    "rows: 3\n"
    "cols: 3\n"
    "board:\n"
    "R01  #1.\n"
    "R02  #21\n"
    "R03  ###\n"
)

WON = (
    "status: won\n"
    "mines_total: 2\n"
    "flags_left: 2\n"
    "rows: 3\n"
    "cols: 3\n"
    "board:\n"
    "R01  #1.\n"
    "R02  121\n"
    "R03  .1#\n"
)

TEXTBOOK = {
    "oracle_version": "1.1.0",
    "seed": 0,
    "attempts": 1,
    "rows": 3,
    "cols": 3,
    "mines_total": 2,
    "first_click": {"row": 1, "col": 3},
    "mines": [{"row": 1, "col": 1}, {"row": 3, "col": 3}],
    "bbbv": 2,
    "max_tier": "enumerate",
    "initial_text": OPENING,
    "gold": [
        {
            "action": {"kind": "open", "at": {"row": 3, "col": 1}},
            "tier": "enumerate",
            "text_after": WON,
        }
    ],
}
