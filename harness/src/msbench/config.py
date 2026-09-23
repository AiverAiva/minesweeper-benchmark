"""Run settings.

Local values come from harness/.env. A variable already set in the process
environment stays as it is. The API key is MS_API_KEY in that file, never a flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    base: str
    model: str
    protocol: str
    temperature: float
    max_retries: int
    samples: int
    timeout: float
    track: str
    binary: str | None


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dotenv() -> Path:
    return project_dir() / ".env"


def api_key() -> str:
    load_dotenv()
    return os.environ.get("MS_API_KEY", "")


def load_dotenv() -> None:
    """Fill unset variables from the env file. Existing process values stay."""
    for path in dotenv_paths():
        if not path.is_file():
            continue
        for key, value in parse_dotenv(path.read_text(encoding="utf-8-sig")):
            os.environ.setdefault(key, value)


def dotenv_paths() -> list[Path]:
    if "MS_ENV_FILE" in os.environ:
        raw = os.environ["MS_ENV_FILE"].strip()
        if not raw:
            return []
        return [Path(raw)]
    return [default_dotenv()]


def parse_dotenv(text: str) -> list[tuple[str, str]]:
    pairs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _env_name(key):
            continue
        pairs.append((key, _unquote(value.strip())))
    return pairs


def load_settings(args) -> Settings:
    load_dotenv()
    temperature = _float(args, "temperature", "MS_TEMPERATURE", 0.0)
    samples = _int(args, "samples", "MS_SAMPLES", 1)
    if temperature < 0:
        raise ConfigError("temperature 需要大於或等於 0。")
    if samples < 1:
        raise ConfigError("samples 需要至少 1。")
    if temperature == 0 and samples != 1:
        raise ConfigError("temperature 0 每個位置只抽一次。要重複抽樣就調高 MS_TEMPERATURE。")
    max_retries = _int(args, "max_retries", "MS_MAX_RETRIES", 2)
    timeout = _float(args, "timeout", "MS_TIMEOUT", 120.0)
    if max_retries < 0:
        raise ConfigError("max_retries 需要大於或等於 0。")
    if timeout <= 0:
        raise ConfigError("timeout 需要大於 0。")
    base = _text(args, "base", "MS_API_BASE", "")
    return Settings(
        base=base.rstrip("/"),
        model=_text(args, "model", "MS_MODEL", ""),
        protocol=_text(args, "protocol", "MS_PROTOCOL", "instruct"),
        temperature=temperature,
        max_retries=max_retries,
        samples=samples,
        timeout=timeout,
        track=_text(args, "track", "MS_TRACK", "adhoc"),
        binary=_optional_text(args, "binary", "MSENGINE"),
    )


def require_api(settings: Settings, key: str) -> None:
    env_file = default_dotenv()
    if not settings.base:
        raise ConfigError(f"在 {env_file} 設定 MS_API_BASE，或傳 --base。位址是以 /v1 結尾的 API 根。")
    if not settings.base.startswith(("http://", "https://")):
        raise ConfigError("MS_API_BASE 需要以 http:// 或 https:// 開頭。")
    if not settings.model:
        raise ConfigError(f"在 {env_file} 設定 MS_MODEL，或傳 --model。")
    if not key:
        raise ConfigError(f"在 {env_file} 寫上 MS_API_KEY。")


def _env_name(key: str) -> bool:
    if not key or key[0].isdigit():
        return False
    return all(char.isalnum() or char == "_" for char in key)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def _text(args, name: str, env_name: str, default: str) -> str:
    value = _optional_text(args, name, env_name)
    return default if value is None else value


def _optional_text(args, name: str, env_name: str) -> str | None:
    flag = getattr(args, name, None)
    if flag is not None and str(flag) != "":
        return str(flag)
    env = os.environ.get(env_name)
    if env:
        return env
    return None


def _int(args, name: str, env_name: str, default: int) -> int:
    raw = _text(args, name, env_name, str(default))
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigError(f"{env_name} 需要整數，收到 {raw}。") from error


def _float(args, name: str, env_name: str, default: float) -> float:
    raw = _text(args, name, env_name, str(default))
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigError(f"{env_name} 需要數字，收到 {raw}。") from error
