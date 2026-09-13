"""Log what the browser actually did. One file, one function.

Hooked into `Tools.act()`, which every browser-use action passes through.
Records the action name, its params, the element it touched, and a
fingerprint of the page afterwards. What each action should become in an
Optexity automation is the compiler's problem, not this file's.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

PATH = Path(os.environ.get("ACTION_LOG", "run_logs/actions.jsonl"))

# A re-render is fast but not instant - saucedemo renames its add-to-cart
# button ~18ms after the click. Probe immediately and you read the stale DOM
# and log a working click as dead. Learning only; replay never calls this.
SETTLE = float(os.environ.get("ACTION_LOG_SETTLE", "0.3"))

ATTRS = ("id", "name", "type", "placeholder", "aria-label", "href", "role",
         "value", "data-testid", "data-test")


async def log_action(name: str, params: dict, node, browser_session, result=None):
    """Append one executed action. Never raises."""
    try:
        if getattr(result, "error", None):
            return  # only successful actions are replayable

        #if SETTLE:
        #    await asyncio.sleep(SETTLE)
        try:
            url = await browser_session.get_current_page_url()
        except Exception:  
            url = ""

        raw = getattr(node, "attributes", None) or {}
        record = {
            "action": name,
            "params": {k: v for k, v in params.items() if k != "_node"},
            "url": url,
            "tag": (getattr(node, "node_name", None)
                    or getattr(node, "tag_name", "") or "").lower(),
            "attributes": {k: str(v) for k, v in raw.items() if k in ATTRS and v},
            "xpath": getattr(node, "xpath", "") or "",
            "label": label(node),
        }
        # Fingerprint of the page after the action. Two consecutive records
        # with the same hash mean the second one changed nothing. Empty means
        # the probe failed - read that as "unknown", never as "no change".
       # record["after"] = await fingerprint(browser_session, url)

        PATH.parent.mkdir(parents=True, exist_ok=True)
        with PATH.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception: 
        pass


async def fingerprint(browser_session, url: str) -> str:
    """url + title + interactive element count, hashed.

    Structural signals survive re-render noise; raw text length does not.
    Degrades to url-only if the state summary is unavailable, which still
    catches navigation.
    """
    parts = [url]
    fn = getattr(browser_session, "get_browser_state_summary", None)
    if callable(fn):
        try:
            state = await fn()
            parts += [str(getattr(state, "title", "")),
                      str(len(getattr(state, "selector_map", None) or {}))]
        except Exception:  # noqa: BLE001
            pass
    return (hashlib.blake2s("|".join(parts).encode(), digest_size=8).hexdigest()
            if url else "")


def label(node, hops: int = 4) -> str:
    """Visible name of the element. Without it the compiler cannot tell the
    City box from the State box when both accept 'SF'."""
    attrs = getattr(node, "attributes", None) or {}
    for key in ("aria-label", "placeholder", "title"):
        if attrs.get(key):
            return str(attrs[key]).strip()[:80]
    for _ in range(hops):
        if node is None:
            break
        fn = getattr(node, "get_all_children_text", None)
        if callable(fn):
            try:
                text = " ".join(str(fn() or "").split())
                if text:
                    return text[:80]
            except Exception:  # noqa: BLE001
                pass
        node = getattr(node, "parent_node", None) or getattr(node, "parent", None)
    return ""
