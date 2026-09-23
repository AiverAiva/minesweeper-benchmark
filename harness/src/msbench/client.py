"""OpenAI-compatible chat completions.

The API key is read by the caller from MS_API_KEY and passed in. It is not
written into attempt logs. Format failures are model output: this client does
not retry them. Timeouts, connection errors, 429, and 5xx are retried.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

TOKEN_FIELDS = ("max_tokens", "max_completion_tokens")


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
    max_tokens: int = 512,
    temperature: float = 0,
    max_retries: int = 2,
    timeout: float = 120,
    token_field: str = "max_tokens",
    post=None,
    sleep=None,
) -> Completion:
    if token_field not in TOKEN_FIELDS:
        raise ValueError(f"token_field must be one of {TOKEN_FIELDS}")
    sender = urllib_post if post is None else post
    pause = time.sleep if sleep is None else sleep
    url = chat_url(base)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        token_field: max_tokens,
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
        text, finish_reason, input_tokens, output_tokens, reasoning_tokens = parsed
        attempts.append(
            _attempt(
                index,
                http_status=status,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        )
        return Completion(text=text, attempts=attempts, transport_error=None)
    return Completion(text="", attempts=attempts, transport_error=last_error)


def urllib_post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def _parse_success(raw: str) -> tuple[str, str | None, int | None, int | None, int | None] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    text = join_message(message)
    if text is None:
        return None
    finish_reason = choices[0].get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    input_tokens, output_tokens, reasoning_tokens = usage_tokens(payload)
    return text, finish_reason, input_tokens, output_tokens, reasoning_tokens


def join_message(message: dict) -> str | None:
    content = as_text(message.get("content"))
    reasoning = as_text(message.get("reasoning_content"))
    if content is None or reasoning is None:
        return None
    if content and reasoning:
        return f"{content}\n{reasoning}"
    return content or reasoning


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


def usage_tokens(payload: dict) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    details = usage.get("completion_tokens_details")
    reasoning = None
    if isinstance(details, dict) and "reasoning_tokens" in details:
        reasoning = _optional_int(details.get("reasoning_tokens"))
    return (
        _optional_int(usage.get("prompt_tokens")),
        _optional_int(usage.get("completion_tokens")),
        reasoning,
    )


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
        "error": error,
    }


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def clip(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]
