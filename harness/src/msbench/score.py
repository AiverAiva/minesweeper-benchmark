"""Five-way label for one model reply.

Order: exactly one tool call, engine rejection, membership in the live forced
set, then the result of applying anything else. A forced action stays supported
even when the board ends.
"""

from __future__ import annotations

from msbench.engine import EngineError, EngineSession

LABELS = ("supported", "lucky", "mine", "illegal", "format")
LEGAL = ("supported", "lucky", "mine")


class GoldError(RuntimeError):
    pass


def hidden_count(board_text: str) -> int:
    lines = board_text.splitlines()
    try:
        start = lines.index("board:") + 1
    except ValueError:
        return 0
    return sum(line.count("#") for line in lines[start:])


def classify(session: EngineSession, action: dict | None, judged: dict | None = None) -> dict:
    if judged is None:
        judged = session.forced()
    actions = judged.get("actions")
    if not isinstance(actions, list):
        raise EngineError("protocol", "forced 回應沒有 actions")
    outcome = {
        "forced_tier": judged.get("tier"),
        "forced_actions": actions,
        "forced_count": len(actions),
        "forced_open_count": sum(1 for action in actions if action.get("kind") == "open"),
        "action": None,
        "status_after": None,
        "text_after": None,
        "detail": None,
        "label": "format",
    }
    parsed = action
    if parsed is None:
        return outcome
    outcome["action"] = parsed
    if parsed in actions:
        applied = session.apply(parsed)
        if not applied.get("ok"):
            raise EngineError(
                str(applied.get("error") or "apply"),
                f"必走動作無法執行: {applied.get('detail') or parsed}",
            )
        outcome["label"] = "supported"
        outcome["status_after"] = applied.get("status")
        outcome["text_after"] = applied.get("text")
        return outcome
    applied = session.apply(parsed)
    if not applied.get("ok"):
        if applied.get("error") == "illegal":
            outcome["label"] = "illegal"
            outcome["detail"] = applied.get("detail")
            return outcome
        raise EngineError(str(applied.get("error") or "apply"), str(applied.get("detail") or ""))
    outcome["status_after"] = applied.get("status")
    outcome["text_after"] = applied.get("text")
    outcome["label"] = "mine" if applied.get("status") == "lost" else "lucky"
    return outcome


def replay_gold(session: EngineSession, fixture: dict) -> None:
    opened = session.new_fixture(fixture)
    if opened.get("text") != fixture.get("initial_text"):
        raise GoldError("開局畫面與 fixture.initial_text 不一致")
    gold = fixture.get("gold") or []
    if not gold:
        raise GoldError("fixture 沒有 gold 步")
    last = opened
    for index, step in enumerate(gold, start=1):
        judged = session.forced()
        if judged.get("tier") != step.get("tier"):
            raise GoldError(f"第 {index} 步層級是 {judged.get('tier')}，fixture 寫 {step.get('tier')}")
        if step.get("action") not in judged.get("actions", []):
            raise GoldError(f"第 {index} 步不在當下的必走集合")
        last = session.apply(step["action"])
        if not last.get("ok"):
            raise GoldError(f"第 {index} 步無法執行: {last.get('detail')}")
        if last.get("text") != step.get("text_after"):
            raise GoldError(f"第 {index} 步畫面與 text_after 不一致")
    if last.get("status") != "won":
        raise GoldError(f"gold 走完的狀態是 {last.get('status')}")
