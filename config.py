"""CLI harness configuration. Load config from a JSON file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    context_limit: int = 128000


@dataclass(frozen=True, slots=True)
class Config:
    llm: LLMConfig
    language_servers: dict = field(default_factory=dict)
    debug_adapters: dict = field(default_factory=dict)
    test_runners: dict = field(default_factory=dict)
    compaction: dict = field(default_factory=dict)
    git: dict = field(default_factory=dict)


_REQUIRED_LLM_KEYS = ('base_url', 'api_key', 'model')


def load(path: str | Path = 'config.json') -> Config:
    """Load and validate the harness config from *path*."""
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"config file not found at '{path.resolve()}' — create config.json with your LLM settings")
    data = json.loads(text)

    raw_llm = data.get('llm')
    if not isinstance(raw_llm, dict):
        raise ValueError("'llm' block is missing or not a mapping")

    llm_data: dict[str, object] = {}
    for key in _REQUIRED_LLM_KEYS:
        if key not in raw_llm:
            raise ValueError(f"missing required 'llm.{key}'")
        llm_data[key] = raw_llm[key]

    EXTRA_LLM_KEYS = ('temperature', 'max_tokens', 'context_limit')
    for key in EXTRA_LLM_KEYS:
        if key in raw_llm:
            llm_data[key] = raw_llm[key]

    llm = LLMConfig(**llm_data)

    future_blocks = ('language_servers', 'debug_adapters', 'test_runners', 'compaction', 'git')
    passthru: dict[str, dict] = {}
    for k in future_blocks:
        if k in data and isinstance(data[k], dict):
            passthru[k] = data[k]

    return Config(llm=llm, **passthru)
