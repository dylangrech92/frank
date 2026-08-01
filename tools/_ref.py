"""Shared ``data-qa-ref`` locator resolution for element-targeting tools.

Refs are assigned by the ``snapshot`` tool's DOM walker (as ``data-qa-ref``
attributes) and stay valid until the element is removed or the page navigates.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page


def locator_for_ref(page: Page, ref: str) -> Locator | None:
    """Return a Locator for *ref*, or ``None`` when no element currently carries it."""
    locator = page.locator(f'[data-qa-ref="{ref}"]')
    if locator.count() == 0:
        return None
    return locator
