# Browser history action cache

The history compiler turns one Browser Use trajectory into a strict, ordered,
website-agnostic cache that Optexity can use as procedural-memory evidence.

## Pipeline

```text
raw_history.json
  -> schema and action/result alignment
  -> ordered observed steps
  -> typed action adapters
  -> locator candidates from interacted-element evidence
  -> browser_use_action_cache.json
```

The compiler never uses temporary Browser Use element indexes as replay locators.
It builds locator options only from recorded DOM attributes, accessibility names,
and XPath evidence. Unknown or unsafe actions remain visible with an explicit
decision; they are never silently dropped.

## Trust rules

- Action, result, and interacted-element arrays are aligned by their recorded
  positions without truncating unequal batches.
- Failed and provably unexecuted actions stay in the audit but are not promoted.
- Terminal `done` is retained as evidence and excluded from browser replay.
- A negative final Browser Use judge verdict makes the source run unsuccessful
  and withholds all replay candidates.
- Locator scores only rank evidence-derived options. They do not prove uniqueness,
  visibility, actionability, or correctness; Optexity validates those live.
- Extra action fields and schema drift are preserved/sanitized and fail closed.
- Obvious secret-bearing arguments are redacted, but raw history and caches should
  still be treated as sensitive artifacts and never committed.

## Action coverage

The adapter registry currently emits typed candidates for common element actions
(input, click, native select, upload, and targeted scroll) and direct actions such
as navigation, search, wait, back, text finding, and supported key presses.
Observation-only JavaScript can be excluded when proven read-only. Mutating or
unrecognized JavaScript, custom actions, and ambiguous targets remain explicit
agentic/manual work.

This is action-capability coverage, not website hardcoding. The compiler contains
no domain, page URL, website name, credential, or site-specific selector branches.

## Usage

```python
from pathlib import Path

from browser_use.agent.history_compiler import compile_history_to_action_cache

compile_history_to_action_cache(
    Path("raw_history.json"),
    Path("browser_use_action_cache.json"),
    task_instruction="Complete the recorded workflow",
)
```

The output is written atomically with restrictive permissions. Current caches are
draft evidence: candidate locators still require live replay validation before
they can become active workflow memory.

## Verification

```bash
python -m pytest -q tests/ci/test_history_compiler.py
python -m ruff check browser_use/agent/history_compiler.py browser_use/agent/history_cache tests/ci/test_history_compiler.py
python -m pyright browser_use/agent/history_compiler.py browser_use/agent/history_cache
```

The focused tests verify ordered compilation, evidence-derived locators, terminal
handling, and withholding candidates after a failed source-run judge.
