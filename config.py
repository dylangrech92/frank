"""CLI harness configuration. Load config from a JSON file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    context_limit: int = 128000
    stream: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    llm: LLMConfig
    # Optional second provider, used for one thing only: turning a captured
    # screenshot into text (see agent._attach_screenshot). It is a SIBLING of
    # ``llm`` rather than a key inside it because unknown keys inside the ``llm``
    # block are silently dropped by the allow-list below. ``None`` means no
    # vision provider is configured and images go to the main provider exactly as
    # they always have.
    vision: LLMConfig | None = None
    language_servers: dict = field(default_factory=dict)
    debug_adapters: dict = field(default_factory=dict)
    test_runners: dict = field(default_factory=dict)
    compaction: dict = field(default_factory=dict)
    git: dict = field(default_factory=dict)
    linters: dict = field(default_factory=dict)
    subagents: dict = field(default_factory=dict)
    browser: dict = field(default_factory=dict)


_REQUIRED_LLM_KEYS = ('base_url', 'api_key', 'model')
_EXTRA_LLM_KEYS = ('temperature', 'max_tokens', 'context_limit', 'stream')


def _llm_from(raw: object, block: str) -> LLMConfig:
    """Validate one provider block and build its ``LLMConfig``.

    Shared by every provider block (``llm`` and the optional ``vision``) so a
    second provider cannot drift into a second, laxer validator: the same three
    keys are required and the same optional keys are allow-listed for both.
    Keys outside the allow-list are dropped, which is exactly why a new provider
    is a sibling block rather than a key nested inside ``llm``.

    Args:
        raw: The block as parsed from JSON; anything that is not a mapping is
            rejected here rather than surfacing later as a TypeError.
        block: The block's name, used verbatim in error messages so a bad
            ``vision`` block is never reported as a bad ``llm`` block.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"'{block}' block is missing or not a mapping")

    data: dict[str, Any] = {}
    for key in _REQUIRED_LLM_KEYS:
        if key not in raw:
            raise ValueError(f"missing required '{block}.{key}'")
        data[key] = raw[key]

    for key in _EXTRA_LLM_KEYS:
        if key in raw:
            data[key] = raw[key]

    return LLMConfig(**data)


def load(path: str | Path = 'config.json') -> Config:
    """Load and validate the harness config from *path*."""
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"config file not found at '{path.resolve()}' — create config.json with your LLM settings")
    data = json.loads(text)

    llm = _llm_from(data.get('llm'), 'llm')

    # The vision provider is optional: a config without the block parses to
    # vision=None, and every image path behaves exactly as it did before the
    # block existed. Present-but-malformed is still an error — a typo'd vision
    # block must not degrade silently into "no vision configured".
    vision = _llm_from(data['vision'], 'vision') if 'vision' in data else None

    future_blocks = ('language_servers', 'debug_adapters', 'test_runners', 'compaction', 'git', 'linters', 'subagents', 'browser')
    passthru: dict[str, dict] = {}
    for k in future_blocks:
        if k in data and isinstance(data[k], dict):
            passthru[k] = data[k]

    # Expand ~ and ${ENV} in debug-adapter command tokens so machine-specific
    # adapter paths stay out of the committed config (env-var indirection).
    import os as _os
    adapters = passthru.get('debug_adapters')
    if isinstance(adapters, dict):
        for _lang, _entry in adapters.items():
            if isinstance(_entry, dict) and isinstance(_entry.get('command'), list):
                _entry['command'] = [
                    _os.path.expanduser(_os.path.expandvars(_tok)) if isinstance(_tok, str) else _tok
                    for _tok in _entry['command']
                ]

    return Config(llm=llm, vision=vision, **passthru)
