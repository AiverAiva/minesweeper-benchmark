"""Six-way label for one model reply.

Order: exactly one tool call, engine rejection, a lost game, a flag placed on
a cell that is not a mine, membership in the exact safe or mine set, then
lucky. The tiered oracle is stored beside the label and does not decide it.
"""

from __future__ import annotations

from msbench.engine import EngineError, EngineSession

LABELS = ("supported", "lucky", "mine", "misflag", "illegal", "format")
LEGAL = ("supported", "lucky", "mine", "misflag")


class GoldError(RuntimeError):
    pass


def hidden_count(board_text: str) -> int:
    lines = board_text.splitlines()
    try:
        start = lines.index("board:") + 1
    except ValueError:
        return 0
    return sum(line.count("#") for line in lines[start:])


def classify(
    session: EngineSession,
    action: dict | None,
    judged: dict | None = None,
    proven: dict | None = None,
) -> dict:
    if judged is None:
        judged = session.forced()
    if proven is None:
        proven = session.exact()
    actions = judged.get("actions")
    if not isinstance(actions, list):
        raise EngineError("protocol", "engine forced response has no actions list")
    exact_actions = proven.get("actions")
    if not isinstance(exact_actions, list):
        raise EngineError("protocol", "engine exact response has no actions list")
    outcome = {
        "forced_tier": judged.get("tier"),
        "forced_actions": actions,
        "forced_count": len(actions),
        "forced_open_count": sum(1 for item in actions if item.get("kind") == "open"),
        "exact_actions": exact_actions,
        "exact_open_count": sum(1 for item in exact_actions if item.get("kind") == "open"),
        "tier_needed": tier_needed(proven, action),
        "action": None,
        "status_after": None,
        "text_after": None,
        "detail": None,
        "label": "format",
    }
    if action is None:
        return outcome
    outcome["action"] = action
    supported = _is_supported(exact_actions, action)
    applied = session.apply(action)
    if not applied.get("ok"):
        if supported:
            raise EngineError(
                str(applied.get("error") or "apply"),
                f"exact action failed to apply: {applied.get('detail') or action}",
            )
        if applied.get("error") == "illegal":
            outcome["label"] = "illegal"
            outcome["detail"] = applied.get("detail")
            return outcome
        raise EngineError(str(applied.get("error") or "apply"), str(applied.get("detail") or ""))
    outcome["status_after"] = applied.get("status")
    outcome["text_after"] = applied.get("text")
    if applied.get("status") == "lost":
        outcome["label"] = "mine"
        return outcome
    placed_flag = action.get("kind") == "flag" and applied.get("flag_placed")
    if placed_flag and not applied.get("cell_is_mine"):
        outcome["label"] = "misflag"
        return outcome
    outcome["label"] = "supported" if supported else "lucky"
    return outcome


def tier_needed(proven: dict, action: dict | None) -> str | None:
    """Shallowest tier that proves the chosen cell, or None when it is not proved."""
    if not action:
        return None
    wanted = _action_key(action)
    for item in proven.get("actions") or []:
        if _action_key(item) == wanted and item.get("tier"):
            return item.get("tier")
    if action.get("kind") not in ("open", "flag"):
        return None
    at = action.get("at") or {}
    for cell in proven.get("cells") or []:
        if cell.get("row") == at.get("row") and cell.get("col") == at.get("col"):
            return cell.get("tier")
    return None


def _is_supported(actions: list, action: dict) -> bool:
    wanted = _action_key(action)
    return any(_action_key(item) == wanted for item in actions)


def _action_key(action: dict) -> tuple:
    at = action.get("at") or {}
    return (action.get("kind"), at.get("row"), at.get("col"))


def replay_gold(session: EngineSession, fixture: dict) -> None:
    opened = session.new_fixture(fixture)
    if opened.get("text") != fixture.get("initial_text"):
        raise GoldError("opening board differs from fixture.initial_text")
    gold = fixture.get("gold") or []
    if not gold:
        raise GoldError("fixture has no gold steps")
    last = opened
    for index, step in enumerate(gold, start=1):
        judged = session.forced()
        if judged.get("tier") != step.get("tier"):
            raise GoldError(
                f"gold step {index}: engine tier is {judged.get('tier')}, "
                f"fixture says {step.get('tier')}"
            )
        if step.get("action") not in judged.get("actions", []):
            raise GoldError(f"gold step {index} is not in the current forced set")
        last = session.apply(step["action"])
        if not last.get("ok"):
            raise GoldError(f"gold step {index} failed to apply: {last.get('detail')}")
        if last.get("text") != step.get("text_after"):
            raise GoldError(f"gold step {index} board differs from text_after")
    if last.get("status") != "won":
        raise GoldError(f"status after gold is {last.get('status')}, expected won")
