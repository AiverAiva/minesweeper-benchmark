import json
import os
from types import SimpleNamespace

import pytest

from msbench.client import USER_AGENT, FatalApiError, complete_chat, parse_tool_call
from msbench.config import ConfigError, api_key, load_settings


def _settings(**overrides):
    values = {
        "base": None,
        "model": None,
        "protocol": None,
        "temperature": None,
        "max_retries": None,
        "samples": None,
        "timeout": None,
        "track": None,
        "binary": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _clear_env(monkeypatch):
    for name in (
        "MS_API_BASE",
        "MS_API_KEY",
        "MS_MODEL",
        "MS_PROTOCOL",
        "MS_TEMPERATURE",
        "MS_MAX_RETRIES",
        "MS_SAMPLES",
        "MS_TIMEOUT",
        "MS_TRACK",
        "MSENGINE",
    ):
        monkeypatch.delenv(name, raising=False)


def _call(kind="open", row=1, col=1, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "minesweeper_move",
            "arguments": json.dumps({"kind": kind, "row": row, "col": col}),
        },
    }


def _chunk(delta=None, finish=None, usage=None):
    choice = {}
    if delta is not None:
        choice["delta"] = delta
    if finish is not None:
        choice["finish_reason"] = finish
    payload = {"choices": [choice]}
    if usage is not None:
        payload["usage"] = usage
    return "data: " + json.dumps(payload)


def _ok(content="noted", calls=None, reasoning=None, reasoning_tokens=None, finish="stop", split_args=False):
    lines = []
    if content is not None:
        lines.append(_chunk({"content": content}))
    if reasoning is not None:
        lines.append(_chunk({"reasoning_content": reasoning}))
    for call in calls if calls is not None else [_call()]:
        if split_args:
            func = call["function"]
            args = func["arguments"]
            half = len(args) // 2
            first = {"function": {"name": func["name"], "arguments": args[:half]}}
            second = {"function": {"arguments": args[half:]}}
            lines.append(_chunk({"tool_calls": [{**call, **first}]}))
            lines.append(_chunk({"tool_calls": [{**call, **second}]}))
        else:
            lines.append(_chunk({"tool_calls": [call]}))
    usage = {"prompt_tokens": 3, "completion_tokens": 4}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    lines.append(_chunk({}, finish=finish, usage=usage))
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


def test_flag_overrides_env_and_the_key_stays_out_of_settings(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MS_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("MS_MODEL", "env-model")
    monkeypatch.setenv("MS_API_KEY", "sk-secret")
    settings = load_settings(_settings(base="https://flag.example/v1/", model="flag-model"))
    assert settings.base == "https://flag.example/v1"
    assert settings.model == "flag-model"
    assert api_key() == "sk-secret"
    assert not hasattr(settings, "key")


def test_dotenv_fills_unset_values_and_keeps_a_shell_override(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    # setdefault writes straight into os.environ. delenv only records a key
    # that already exists, so file-filled names would leak into later tests.
    introduced = ("MS_API_BASE", "MS_API_KEY", "MS_PROTOCOL")
    saved = {name: os.environ.get(name) for name in introduced}
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "export MS_API_BASE=https://file.example/v1",
                "MS_MODEL=file-model",
                'MS_API_KEY="sk-from-file"',
                "MS_PROTOCOL=thinking # column label",
                "not a pair",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MS_ENV_FILE", str(env_file))
    monkeypatch.setenv("MS_MODEL", "shell-model")
    try:
        settings = load_settings(_settings())
        assert settings.base == "https://file.example/v1"
        assert settings.model == "shell-model"
        assert settings.protocol == "thinking"
        assert api_key() == "sk-from-file"
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def test_temperature_zero_rejects_repeated_samples(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(ConfigError):
        load_settings(_settings(temperature=0, samples=3))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(status):
    calls = {"n": 0}

    def post(url, headers, body, timeout):
        calls["n"] += 1
        assert url == "https://example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["User-Agent"] == USER_AGENT
        assert body["tools"][0]["function"]["name"] == "minesweeper_move"
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "minesweeper_move"},
        }
        assert body["stream"] is True
        assert body["messages"] == [{"role": "user", "content": "board"}]
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
        return status, "nope"

    with pytest.raises(FatalApiError):
        complete_chat(
            prompt="board",
            base="https://example.com/v1",
            key="sk-test",
            model="m",
            post=post,
            sleep=lambda _seconds: None,
        )
    assert calls["n"] == 1


def test_server_error_is_retried_then_scored():
    calls = {"n": 0}
    slept = []

    def post(url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, "busy"
        return 200, _ok(reasoning_tokens=2)

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1/chat/completions",
        key="sk-super-secret",
        model="m",
        max_retries=2,
        post=post,
        sleep=slept.append,
    )
    assert calls["n"] == 2
    assert slept == [1.0]
    assert result.transport_error is None
    assert result.text == "noted"
    assert result.action == {"kind": "open", "at": {"row": 1, "col": 1}}
    assert result.attempts[1]["reasoning_tokens"] == 2
    assert result.attempts[1]["total_tokens"] == 7
    assert result.attempts[0]["http_status"] == 500
    assert "sk-super-secret" not in json.dumps(result.attempts)


def test_retryable_failures_are_reported_before_the_retry():
    calls = {"n": 0}
    notes: list[str] = []

    def post(url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, "busy"
        return 200, _ok()

    complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        max_retries=1,
        post=post,
        sleep=lambda _seconds: None,
        report=notes.append,
    )
    assert calls["n"] == 2
    assert len(notes) == 1
    assert "api attempt 1 failed" in notes[0]
    assert "http 500" in notes[0]
    assert "retrying in 1s" in notes[0]


def test_fatal_client_errors_are_not_reported_as_retries():
    notes: list[str] = []

    def post(url, headers, body, timeout):
        return 400, "nope"

    with pytest.raises(FatalApiError):
        complete_chat(
            prompt="board",
            base="https://example.com/v1",
            key="sk-test",
            model="m",
            post=post,
            sleep=lambda _seconds: None,
            report=notes.append,
        )
    assert notes == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_statuses_become_transport_when_retries_are_spent(status):
    def post(url, headers, body, timeout):
        return status, "later"

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        max_retries=0,
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.transport_error is not None
    assert result.attempts[0]["http_status"] == status


def test_completion_url_is_not_doubled():
    seen = {}

    def post(url, headers, body, timeout):
        seen["url"] = url
        return 200, _ok()

    complete_chat(
        prompt="board",
        base="https://example.com/v1/chat/completions",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert seen["url"] == "https://example.com/v1/chat/completions"


def test_plain_json_body_still_parses_when_stream_is_ignored():
    payload = {
        "choices": [
            {
                "message": {"content": "noted", "tool_calls": [_call(kind="flag", row=2, col=3)]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }

    def post(url, headers, body, timeout):
        assert body["stream"] is True
        return 200, json.dumps(payload)

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.attempts[0]["total_tokens"] == 16
    assert result.attempts[0]["reasoning_tokens"] == 4
    assert result.action == {"kind": "flag", "at": {"row": 2, "col": 3}}


def test_split_tool_arguments_reassemble():
    def post(url, headers, body, timeout):
        return 200, _ok(calls=[_call(kind="flag", row=2, col=3)], split_args=True)

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.action == {"kind": "flag", "at": {"row": 2, "col": 3}}
    assert result.transport_error is None


def test_keepalive_comments_are_ignored():
    body = ": keep-alive\n\n" + _ok() + ": ping\n\n"

    def post(url, headers, body_, timeout):
        return 200, body

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.action == {"kind": "open", "at": {"row": 1, "col": 1}}
    assert result.transport_error is None


def test_a_body_without_a_call_is_returned_once():
    calls = {"n": 0}

    def post(url, headers, body, timeout):
        calls["n"] += 1
        return 200, _ok(content="I pass", calls=[], reasoning_tokens=None)

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert calls["n"] == 1
    assert result.transport_error is None
    assert result.text == "I pass"
    assert result.action is None
    assert result.attempts[0]["reasoning_tokens"] is None


def test_two_calls_yield_no_action():
    def post(url, headers, body, timeout):
        return 200, _ok(calls=[_call(kind="open", row=1, col=1), _call(kind="flag", row=2, col=2, call_id="call_2")])

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.action is None
    assert result.transport_error is None


def test_one_call_beside_reasoning_text_parses():
    def post(url, headers, body, timeout):
        return 200, _ok(content="working", calls=[_call(kind="open", row=2, col=2)], reasoning="hmm")

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.action == {"kind": "open", "at": {"row": 2, "col": 2}}
    assert result.text == "working"
    assert result.reasoning == "hmm"


def test_broken_call_arguments_yield_no_action():
    bad = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "minesweeper_move", "arguments": "not json {"},
    }

    def post(url, headers, body, timeout):
        return 200, _ok(calls=[bad])

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.action is None
    assert result.transport_error is None


def test_direct_tool_call_parsing_rejects_bad_shapes():
    good = {"tool_calls": [_call(kind="chord", row=3, col=4)]}
    assert parse_tool_call(good) == {"kind": "chord", "at": {"row": 3, "col": 4}}
    assert parse_tool_call({"tool_calls": []}) is None
    assert parse_tool_call({}) is None
    assert parse_tool_call({"tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]}) is None
    assert parse_tool_call({"tool_calls": [_call(kind="dig", row=1, col=1)]}) is None
    assert parse_tool_call({"tool_calls": [_call(kind="open", row=0, col=1)]}) is None


def test_error_bodies_are_truncated():
    def post(url, headers, body, timeout):
        return 500, "x" * 2000

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        max_retries=0,
        post=post,
        sleep=lambda _seconds: None,
    )
    assert result.transport_error is not None
    assert len(result.attempts[0]["error"]) <= 500
    assert len(result.attempts) == 1
