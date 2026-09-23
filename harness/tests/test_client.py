import json
import os
from types import SimpleNamespace

import pytest

from msbench.client import FatalApiError, complete_chat
from msbench.config import ConfigError, api_key, load_settings
from msbench.parse import parse_action


def _settings(**overrides):
    values = {
        "base": None,
        "model": None,
        "protocol": None,
        "max_tokens": None,
        "temperature": None,
        "max_retries": None,
        "samples": None,
        "token_field": None,
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
        "MS_MAX_TOKENS",
        "MS_TEMPERATURE",
        "MS_MAX_RETRIES",
        "MS_SAMPLES",
        "MS_TOKEN_FIELD",
        "MS_TIMEOUT",
        "MS_TRACK",
        "MSENGINE",
    ):
        monkeypatch.delenv(name, raising=False)


def _ok(content="ACTION: open row=1 col=1", reasoning=None, reasoning_tokens=None, finish="stop"):
    message = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    usage = {"prompt_tokens": 3, "completion_tokens": 4}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return json.dumps(
        {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}
    )


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
        assert "tools" not in body
        assert body["messages"] == [{"role": "user", "content": "board"}]
        assert body["max_tokens"] == 512
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
    assert result.text == "ACTION: open row=1 col=1"
    assert result.attempts[1]["reasoning_tokens"] == 2
    assert result.attempts[0]["http_status"] == 500
    assert "sk-super-secret" not in json.dumps(result.attempts)


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


def test_token_field_switch_sends_only_one_limit():
    seen = {}

    def post(url, headers, body, timeout):
        seen["body"] = body
        return 200, _ok()

    complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        max_tokens=32,
        token_field="max_completion_tokens",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert seen["body"]["max_completion_tokens"] == 32
    assert "max_tokens" not in seen["body"]


def test_a_body_without_an_action_is_returned_once():
    calls = {"n": 0}

    def post(url, headers, body, timeout):
        calls["n"] += 1
        return 200, _ok(content="I pass", reasoning_tokens=None)

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
    assert result.attempts[0]["reasoning_tokens"] is None


def test_reasoning_content_is_joined_and_two_actions_fail_parse():
    def post(url, headers, body, timeout):
        return 200, _ok(content="ACTION: open row=1 col=1", reasoning="ACTION: flag row=2 col=2")

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert parse_action(result.text) is None
    assert "ACTION: open row=1 col=1" in result.text
    assert "ACTION: flag row=2 col=2" in result.text


def test_one_action_in_reasoning_content_parses():
    def post(url, headers, body, timeout):
        return 200, _ok(content="working", reasoning="ACTION: open row=2 col=2")

    result = complete_chat(
        prompt="board",
        base="https://example.com/v1",
        key="sk-test",
        model="m",
        post=post,
        sleep=lambda _seconds: None,
    )
    assert parse_action(result.text) == {"kind": "open", "at": {"row": 2, "col": 2}}


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
