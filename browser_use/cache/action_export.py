"""Export a completed agent run into a deterministic, replayable action log.

This turns an `AgentHistoryList` (LLM-driven exploration, keyed by ephemeral
bracket indices) into a JSON list of stable per-action records - one per
executed action - that a later Playwright locator builder can consume without
ever re-running the LLM. See `optexity_cursor_plan.md` Phase 1/T1.

Nothing here mutates or depends on a live browser/page; it only reads the
in-memory history object returned by `agent.run()`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from browser_use.agent.views import AgentHistoryList
from browser_use.dom.views import DOMInteractedElement

logger = logging.getLogger(__name__)

# ActionModel field name (the single key left after exclude_unset=True) -> our
# stable action_type vocabulary. Anything not listed here is "other" (thinking,
# extraction, screenshot, done, scroll, etc. - nothing a Playwright locator
# needs to replay).
_ACTION_TYPE_BY_KEY = {
	'input': 'input_text',
	'click': 'click',
	'select_dropdown': 'select',
	'navigate': 'navigate',
}

# Param name holding the human-entered value, per action key (used for
# typed_value and for password-likeness checks). Only set for actions where
# that value is meaningful to replay.
_VALUE_PARAM_BY_KEY = {
	'input': 'text',
	'select_dropdown': 'text',
	'navigate': 'url',
}

_PASSWORD_PATTERN = re.compile(r'pass(word)?|pwd', re.IGNORECASE)
_MASKED_VALUE = '***MASKED***'


@dataclass
class ExportedElement:
	tag: str | None
	id: str | None
	name: str | None
	placeholder: str | None
	aria_label: str | None
	role: str | None
	visible_text: str | None
	data_attrs: dict[str, str]
	x_path: str | None
	in_shadow_dom: bool
	element_hash: str | None


@dataclass
class ExportedAction:
	step_index: int
	action_type: str
	typed_value: str | None
	success: bool
	element: ExportedElement | None
	caused_navigation: bool
	raw: dict[str, Any]


def _action_key_and_params(action) -> tuple[str, dict[str, Any]]:
	"""The single set action name and its params, e.g. ('input', {'index': 5, 'text': 'x'})."""
	dumped = action.model_dump(exclude_unset=True, mode='json')
	if not dumped:
		return 'other', {}
	key = next(iter(dumped))
	return key, dumped[key] or {}


def _is_password_like(element: DOMInteractedElement | None, action_key: str) -> bool:
	if action_key != 'input' or element is None:
		return False
	attrs = element.attributes or {}
	if attrs.get('type', '').lower() == 'password':
		return True
	candidates = [attrs.get('name'), attrs.get('id'), attrs.get('placeholder'), attrs.get('aria-label'), element.ax_name]
	return any(candidate and _PASSWORD_PATTERN.search(candidate) for candidate in candidates)


def _element_hash_str(element: DOMInteractedElement) -> str:
	# stable_hash filters transient CSS classes (focus/hover/etc.), so prefer it;
	# element_hash is the fallback for elements where that couldn't be computed.
	value = element.stable_hash if element.stable_hash is not None else element.element_hash
	return f'{value:016x}'


def _build_element(element: DOMInteractedElement | None) -> ExportedElement | None:
	if element is None:
		return None
	attrs = element.attributes or {}
	return ExportedElement(
		tag=element.node_name.lower() if element.node_name else None,
		id=attrs.get('id'),
		name=attrs.get('name'),
		placeholder=attrs.get('placeholder'),
		aria_label=attrs.get('aria-label') or element.ax_name,
		role=attrs.get('role'),
		visible_text=element.ax_name,
		data_attrs={k: v for k, v in attrs.items() if k.startswith('data-')},
		x_path=element.x_path,
		in_shadow_dom=element.in_shadow_dom,
		element_hash=_element_hash_str(element),
	)


def export_agent_history(
	history: AgentHistoryList,
	output_dir: str | Path | None = None,
	timestamp: str | None = None,
) -> tuple[Path, Path]:
	"""Flatten a completed agent run into `cached_run_<timestamp>.json` (+ `_raw.json`).

	Returns (cached_run_path, raw_dump_path). Callers should guard this call
	with the `OPTEXITY_CACHE_EXPORT=1` env var so default behavior is unchanged.
	"""
	out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
	out_dir.mkdir(parents=True, exist_ok=True)
	ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S_%f')

	records: list[dict[str, Any]] = []
	masked_count = 0

	for step_index, item in enumerate(history.history):
		model_output = item.model_output
		if model_output is None:
			continue

		state = item.state
		next_state = history.history[step_index + 1].state if step_index + 1 < len(history.history) else None
		step_caused_navigation = bool(state and next_state and next_state.url != state.url)

		interacted_elements = state.interacted_element if state else []
		results = item.result

		for action_index, action in enumerate(model_output.action):
			# multi_act stops early on failure/navigation, so a later action in the
			# same step may have no corresponding result - it never ran, skip it.
			if action_index >= len(results):
				continue
			result = results[action_index]
			element = interacted_elements[action_index] if action_index < len(interacted_elements) else None

			action_key, params = _action_key_and_params(action)
			action_type = _ACTION_TYPE_BY_KEY.get(action_key, 'other')

			typed_value: str | None = None
			value_param = _VALUE_PARAM_BY_KEY.get(action_key)
			if value_param is not None:
				typed_value = params.get(value_param)

			is_sensitive = _is_password_like(element, action_key)
			if is_sensitive and typed_value is not None:
				typed_value = _MASKED_VALUE
				masked_count += 1

			raw_action = dict(params)
			if is_sensitive and value_param is not None and value_param in raw_action:
				raw_action[value_param] = _MASKED_VALUE

			record = ExportedAction(
				step_index=item.metadata.step_number if item.metadata else step_index + 1,
				action_type=action_type,
				typed_value=typed_value,
				success=result.error is None,
				element=_build_element(element),
				caused_navigation=step_caused_navigation,
				raw={
					'action_key': action_key,
					'action_params': raw_action,
					'result_error': result.error,
					'result_success': result.success,
				},
			)
			records.append(asdict(record))

	cached_run_path = out_dir / f'cached_run_{ts}.json'
	raw_dump_path = out_dir / f'cached_run_{ts}_raw.json'

	with open(cached_run_path, 'w', encoding='utf-8') as f:
		json.dump(records, f, indent=2)

	with open(raw_dump_path, 'w', encoding='utf-8') as f:
		json.dump(history.model_dump(), f, indent=2)

	kept_by_type: dict[str, int] = {}
	for r in records:
		kept_by_type[r['action_type']] = kept_by_type.get(r['action_type'], 0) + 1
	logger.info(
		f'Exported {len(records)} action record(s) to {cached_run_path} '
		f'(by type: {kept_by_type}, masked_values={masked_count}); raw dump at {raw_dump_path}'
	)

	return cached_run_path, raw_dump_path
