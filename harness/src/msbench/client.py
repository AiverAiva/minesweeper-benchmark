"""OpenAI-compatible chat completions over SSE.

The API key is read by the caller from MS_API_KEY and passed in. It is not
written into attempt logs. Format failures are model output: this client does
not retry them. Timeouts, connection errors, 429, and 5xx are retried.

Requests stream (`stream: true`) so long generations keep bytes flowing and
stay under proxy read timeouts. Chunks are reassembled into one text, one
reasoning string, and the tool calls before scoring. A plain JSON body is
still accepted in case a gateway ignores the stream flag.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "msbench/0.1.0"

MOVE_TOOL_NAME = "minesweeper_move"

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": MOVE_TOOL_NAME,
        "description": "Play one minesweeper move. Call exactly once per turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["open", "flag", "chord"]},
                "row": {"type": "integer", "minimum": 1},
                "col": {"type": "integer", "minimum": 1},
            },
            "required": ["kind", "row", "col"],
            "additionalProperties": False,
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": MOVE_TOOL_NAME}}


class FatalApiError(RuntimeError):
    def __init__(self, status: int, detail: str, attempts: list[dict]):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail
        self.attempts = attempts


@dataclass
class Completion:
    text: str
    attempts: list[dict] = field(default_factory=list)
    transport_error: str | None = None
    action: dict | None = None
    reasoning: str = ""


def chat_url(base: str) -> str:
    root = base.strip().rstrip("/")
    suffix = "/chat/completions"
    if root.endswith(suffix):
        return root
    return root + suffix


def backoff_seconds(retry_number: int) -> float:
    return float(min(2 ** (retry_number - 1), 8))


def complete_chat(
    *,
    prompt: str,
    base: str,
    key: str,
    model: str,
    temperature: float = 0,
    max_retries: int = 2,
    timeout: float = 120,
    post=None,
    sleep=None,
) -> Completion:
    sender = urllib_post if post is None else post
    pause = time.sleep if sleep is None else sleep
    url = chat_url(base)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    # This gateway counts reasoning against max_tokens, so the request leaves it unset.
    # Streaming keeps bytes flowing on long generations; proxies time out idle waits.
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [MOVE_TOOL],
        "tool_choice": TOOL_CHOICE,
        "stream": True,
    }
    attempts: list[dict] = []
    last_error = "transport"
    for index in range(max_retries + 1):
        if index:
            pause(backoff_seconds(index))
        started = time.perf_counter()
        try:
            status, raw = sender(url, headers, body, timeout)
        except Exception as error:
            message = clip(f"{type(error).__name__}: {error}")
            attempts.append(_attempt(index, latency_ms=_elapsed(started), error=message))
            last_error = message
            continue
        latency_ms = _elapsed(started)
        if status != 200:
            message = clip(f"http {status}: {raw}")
            attempts.append(
                _attempt(index, http_status=status, latency_ms=latency_ms, error=message)
            )
            if status == 429 or status >= 500:
                last_error = message
                continue
            raise FatalApiError(status, message, attempts)
        parsed = _parse_success(raw)
        if parsed is None:
            message = clip(f"unreadable response: {raw}")
            attempts.append(
                _attempt(index, http_status=status, latency_ms=latency_ms, error=message)
            )
            last_error = message
            continue
        text, reasoning, action, finish_reason, input_tokens, output_tokens, reasoning_tokens, total_tokens = parsed
        attempts.append(
            _attempt(
                index,
                http_status=status,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
            )
        )
        return Completion(text=text, attempts=attempts, transport_error=None, action=action, reasoning=reasoning)
    return Completion(text="", attempts=attempts, transport_error=last_error)


def urllib_post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def _parse_success(
    raw: str,
) -> tuple[str, str, dict | None, str | None, int | None, int | None, int | None, int | None] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
        return _parse_message_payload(payload)
    return _parse_sse(raw)


def _parse_message_payload(
    payload: dict,
) -> tuple[str, str, dict | None, str | None, int | None, int | None, int | None, int | None] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    text = message_text(message)
    reasoning = reasoning_text(message)
    if text is None or reasoning is None:
        return None
    finish_reason = choices[0].get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    input_tokens, output_tokens, reasoning_tokens, total_tokens = usage_tokens(payload)
    return (
        text,
        reasoning,
        parse_tool_call(message),
        finish_reason,
        input_tokens,
        output_tokens,
        reasoning_tokens,
        total_tokens,
    )


def _parse_sse(
    raw: str,
) -> tuple[str, str, dict | None, str | None, int | None, int | None, int | None, int | None] | None:
    """Reassemble an SSE stream into one completion. Keep-alive comments are ignored."""
    texts: list[str] = []
    thoughts: list[str] = []
    calls: dict[int, dict] = {}
    finish_reason: str | None = None
    usage: dict | None = None
    saw_chunk = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        finished = choice.get("finish_reason")
        if isinstance(finished, str):
            finish_reason = finished
        delta = choice.get("delta")
        if delta is None:
            continue
        if not isinstance(delta, dict):
            return None
        saw_chunk = True
        content = as_text(delta.get("content"))
        thought = as_text(delta.get("reasoning_content"))
        if content is None or thought is None:
            return None
        texts.append(content)
        thoughts.append(thought)
        pending = delta.get("tool_calls")
        if pending is None:
            continue
        if not isinstance(pending, list):
            return None
        for item in pending:
            if not isinstance(item, dict):
                return None
            try:
                index = int(item.get("index", 0))
            except (TypeError, ValueError):
                return None
            slot = calls.setdefault(index, {"id": None, "name": None, "args": []})
            if isinstance(item.get("id"), str):
                slot["id"] = slot["id"] or item["id"]
            func = item.get("function")
            if func is None:
                continue
            if not isinstance(func, dict):
                return None
            name = func.get("name")
            if isinstance(name, str) and name:
                slot["name"] = slot["name"] or name
            args = func.get("arguments")
            if isinstance(args, str):
                slot["args"].append(args)
            elif isinstance(args, dict):
                slot["args"].append(json.dumps(args))
            elif args is not None:
                return None
    if not saw_chunk and finish_reason is None and usage is None:
        return None
    message = {
        "tool_calls": [
            {
                "id": slot["id"],
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": "".join(slot["args"]),
                },
            }
            for _, slot in sorted(calls.items())
        ]
    }
    if isinstance(usage, dict):
        input_tokens, output_tokens, reasoning_tokens, total_tokens = usage_tokens(
            {"usage": usage}
        )
    else:
        input_tokens, output_tokens, reasoning_tokens, total_tokens = None, None, None, None
    return (
        "".join(texts),
        "".join(thoughts),
        parse_tool_call(message),
        finish_reason,
        input_tokens,
        output_tokens,
        reasoning_tokens,
        total_tokens,
    )


def parse_tool_call(message: dict) -> dict | None:
    """Return the single minesweeper_move call, or None when it is not exactly one."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, dict):
        return None
    func = call.get("function")
    if not isinstance(func, dict) or func.get("name") != MOVE_TOOL_NAME:
        return None
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    kind = args.get("kind")
    row = args.get("row")
    col = args.get("col")
    if kind not in ("open", "flag", "chord"):
        return None
    if isinstance(row, bool) or isinstance(col, bool):
        return None
    if not isinstance(row, int) or not isinstance(col, int):
        return None
    if row < 1 or col < 1:
        return None
    return {"kind": kind, "at": {"row": row, "col": col}}


def message_text(message: dict) -> str | None:
    return as_text(message.get("content"))


def reasoning_text(message: dict) -> str | None:
    return as_text(message.get("reasoning_content"))


def as_text(value) -> str | None:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                return None
        return "".join(parts)
    return None


def usage_tokens(payload: dict) -> tuple[int | None, int | None, int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None, None
    details = usage.get("completion_tokens_details")
    reasoning = None
    if isinstance(details, dict) and "reasoning_tokens" in details:
        reasoning = _optional_int(details.get("reasoning_tokens"))
    prompt = _optional_int(usage.get("prompt_tokens"))
    completion = _optional_int(usage.get("completion_tokens"))
    total = _optional_int(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return prompt, completion, reasoning, total


def _optional_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _attempt(
    index: int,
    *,
    http_status: int | None = None,
    latency_ms: float | None = None,
    finish_reason: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    total_tokens: int | None = None,
    error: str | None = None,
) -> dict:
    return {
        "attempt": index,
        "http_status": http_status,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "error": error,
    }


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def clip(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]
