"""Loads the configured enrichment plugin by name.

The plugin name comes from `config.ENRICHMENT_PLUGIN` at call time rather
than a static import, so a new plugin module can be dropped into
`relay/plugins/` and enabled by editing configuration alone.
"""
from __future__ import annotations

import importlib
import pkgutil

from relay.errors import PluginLoadError

_PLUGIN_PACKAGE = "relay.plugins"

_cache: dict[str, object] = {}


def list_available_plugin_names() -> list[str]:
    """List module names under `relay/plugins/` that could be passed to
    `load_plugin`. Does not check that each one actually exposes a
    `PLUGIN` entry point - that's only verified at load time."""
    import relay.plugins as plugins_pkg

    return sorted(
        info.name
        for info in pkgutil.iter_modules(plugins_pkg.__path__)
        if info.name != "loader"
    )


def load_plugin(name: str):
    """Import `relay.plugins.<name>` and return its `PLUGIN` entry
    point.

    Successful loads are cached by name for the lifetime of the process,
    so a misconfigured `config.ENRICHMENT_PLUGIN` pays the
    `PluginLoadError` cost on every call, but a working one only imports
    once.

    Raises `PluginLoadError` if the module does not exist or does not
    expose a `PLUGIN` object.
    """
    if name in _cache:
        return _cache[name]

    module_name = f"{_PLUGIN_PACKAGE}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PluginLoadError(f"no such enrichment plugin: {name}") from exc
    try:
        plugin = module.PLUGIN
    except AttributeError as exc:
        raise PluginLoadError(f"plugin {name} has no PLUGIN entry point") from exc

    _cache[name] = plugin
    return plugin


def clear_cache() -> None:
    """Forget every cached plugin. Mainly useful for tools that reload
    plugin modules in place during development."""
    _cache.clear()
