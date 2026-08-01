"""Snapshot tool: the model's default eyes on the page.

Injects a JS DOM walker that tags every interactable/semantic element with a
stable ``data-qa-ref`` attribute and returns a compact, indented text tree —
role, accessible name, state, and ref — for the other tools to target.
"""

from __future__ import annotations

from typing import Any

from runtime.browser import get_session
from tools.base import Tool
from tools.result import ToolResult

# Runs in the page. Refs (data-qa-ref="eN") are assigned once and persist as DOM
# attributes, so they stay valid across repeated snapshot() calls on the same
# page load; a navigation wipes window state (and the DOM), so refs and the
# counter naturally start over. window.__qaGen counts snapshot calls so callers
# can tell how many times the tree has been walked this page load.
_SNAPSHOT_SCRIPT = r"""
() => {
  window.__qaRefCounter = window.__qaRefCounter || 0;
  window.__qaGen = (window.__qaGen || 0) + 1;

  const SEL = 'a,button,input,select,textarea,[role],[onclick],summary,label,' +
    'h1,h2,h3,h4,h5,h6,img[alt]';

  function norm(s) {
    return (s || '').replace(/\s+/g, ' ').trim();
  }

  function isVisible(el) {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    if (parseFloat(style.opacity) === 0) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function roleOf(el) {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return el.hasAttribute('href') ? 'link' : 'generic';
    if (tag === 'button') return 'button';
    if (tag === 'input') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'submit' || type === 'button' || type === 'reset') return 'button';
      return 'textbox';
    }
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'summary') return 'button';
    if (tag === 'label') return 'label';
    if (tag === 'img') return 'img';
    if (/^h[1-6]$/.test(tag)) return 'heading';
    return tag;
  }

  function accName(el) {
    const aria = el.getAttribute('aria-label');
    if (aria) return norm(aria);

    const labelledby = el.getAttribute('aria-labelledby');
    if (labelledby) {
      const parts = labelledby.split(/\s+/).map((id) => {
        const t = document.getElementById(id);
        return t ? norm(t.textContent) : '';
      }).filter(Boolean);
      if (parts.length) return parts.join(' ');
    }

    const tag = el.tagName.toLowerCase();

    if (tag === 'img') return norm(el.getAttribute('alt'));

    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      if (el.id) {
        const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
        if (lbl && norm(lbl.textContent)) return norm(lbl.textContent);
      }
      const parentLabel = el.closest('label');
      if (parentLabel && norm(parentLabel.textContent)) return norm(parentLabel.textContent);
      const type = (el.getAttribute('type') || '').toLowerCase();
      if (tag === 'input' && (type === 'submit' || type === 'button') && el.value) {
        return norm(el.value);
      }
      if (el.placeholder) return norm(el.placeholder);
      return '';
    }

    return norm(el.textContent);
  }

  function stateOf(el) {
    const parts = [];
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();

    if (tag === 'input' && (type === 'checkbox' || type === 'radio')) {
      parts.push(`checked=${!!el.checked}`);
    } else if (tag === 'input' || tag === 'textarea') {
      parts.push(`value=${JSON.stringify(el.value)}`);
    } else if (tag === 'select') {
      parts.push(`value=${JSON.stringify(el.value)}`);
    }

    if ('disabled' in el && el.disabled) parts.push('disabled');
    parts.push(`visible=${isVisible(el)}`);
    return parts.join(' ');
  }

  const lines = [];

  function walk(node, depth) {
    for (const child of node.children) {
      if (child.matches(SEL)) {
        let ref = child.getAttribute('data-qa-ref');
        if (!ref) {
          ref = 'e' + (++window.__qaRefCounter);
          child.setAttribute('data-qa-ref', ref);
        }
        const role = roleOf(child);
        const name = accName(child);
        const state = stateOf(child);
        const indent = '  '.repeat(depth);
        lines.push(`${indent}${role} "${name}" [${ref}] (${state})`);
        walk(child, depth + 1);
      } else {
        walk(child, depth);
      }
    }
  }

  walk(document.body, 0);

  return {
    url: location.href,
    title: document.title,
    generation: window.__qaGen,
    count: lines.length,
    tree: lines.join('\n'),
  };
}
"""


class Snapshot(Tool):
    """Walk the page DOM and return a compact, ref-tagged text tree."""

    name = "snapshot"
    description = (
        "Walk the current page's DOM and return a compact, indented text tree of "
        "every interactable and semantic element (links, buttons, inputs, selects, "
        "textareas, elements with a role or onclick, summary, label, headings, and "
        "images with alt text). Each element is tagged with a stable data-qa-ref "
        "(e.g. 'e3') that other tools use to target it. This is the primary way to "
        "see page state — call it after every navigation or action that might "
        "change the page."
    )
    action = "snapshot the page"
    oversize_hint = "the page has an unusually large number of elements; target a narrower interaction"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the snapshot tool.

        Returns:
            A ``ToolResult`` whose body is the indented ref-tagged text tree
            (prefixed with the page URL and title), or an error when the walk
            fails.
        """
        page = get_session().page

        try:
            state = page.evaluate(_SNAPSHOT_SCRIPT)
        except Exception as exc:
            return ToolResult.err(f"snapshot failed: {exc}", code="snapshot-failed")

        url = state.get("url", "")
        title = state.get("title", "")
        tree = state.get("tree", "")
        body = f"url: {url}\ntitle: {title}\n\n{tree}"

        return ToolResult.ok(
            body,
            url=url,
            title=title,
            element_count=state.get("count", 0),
            generation=state.get("generation", 0),
        )
