"""Filter an exported agent action log down to replayable steps.

Takes the typed records from `action_export.export_agent_history` and drops
exploration noise (non-browser actions, failures, click-to-focus before type,
superseded retries) so Phase 3 can turn the kept list into deterministic
Playwright commands. See `optexity_cursor_plan.md` Phase 2/T2.

Filtering is a pure transform of the record list - no browser, no LLM. The
`StepFilter` protocol lets an LLM-based filter be swapped in later without
touching callers.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Action types that mutate page state and are therefore candidates for replay.
# Everything else (done / extract / scroll / thinking / screenshot / ...) is
# dropped by rule 1 as non-browser.
_MUTATING_ACTION_TYPES = frozenset(
	{'input_text', 'click', 'select', 'navigate', 'upload', 'key_press'}
)


class StepFilter(Protocol):
	"""Pluggable filter over exported action records.

	Implementations must return `(kept, discarded)` where each discarded entry
	is `{"reason": str, "record": <original record>}`. Every decision (kept or
	discarded) must be logged with a reason - hard rule 4.
	"""

	def filter_steps(
		self, records: list[dict[str, Any]]
	) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def _element_hash(record: dict[str, Any]) -> str | None:
	element = record.get('element')
	if not isinstance(element, dict):
		return None
	value = element.get('element_hash')
	return value if isinstance(value, str) and value else None


def _discard(record: dict[str, Any], reason: str, discarded: list[dict[str, Any]]) -> None:
	logger.info(
		f'discarded step_index={record.get("step_index")} '
		f'action_type={record.get("action_type")}: {reason}'
	)
	discarded.append({'reason': reason, 'record': record})


def _keep(record: dict[str, Any], reason: str, kept: list[dict[str, Any]]) -> None:
	logger.info(
		f'kept step_index={record.get("step_index")} '
		f'action_type={record.get("action_type")}: {reason}'
	)
	kept.append(record)


class RuleBasedStepFilter:
	"""Deterministic rule filter (rules 1-5, applied in order)."""

	def filter_steps(
		self, records: list[dict[str, Any]]
	) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
		discarded: list[dict[str, Any]] = []

		# Rule 1: drop non-browser / non-mutating actions.
		after_rule1: list[dict[str, Any]] = []
		for record in records:
			action_type = record.get('action_type')
			if action_type not in _MUTATING_ACTION_TYPES:
				_discard(record, f'non_browser_action:{action_type}', discarded)
			else:
				after_rule1.append(record)

		# Rule 2: drop failed actions.
		after_rule2: list[dict[str, Any]] = []
		for record in after_rule1:
			if record.get('success') is False:
				_discard(record, 'failed_action', discarded)
			else:
				after_rule2.append(record)

		# Rule 3: merge click-to-focus immediately followed by typing into the
		# same element into one input_text step (discard the click). Also merge
		# input_text immediately followed by Enter key_press into one input_text
		# with press_enter=True (search-box submit pattern).
		after_rule3: list[dict[str, Any]] = []
		i = 0
		while i < len(after_rule2):
			current = after_rule2[i]
			nxt = after_rule2[i + 1] if i + 1 < len(after_rule2) else None
			if (
				nxt is not None
				and current.get('action_type') == 'click'
				and nxt.get('action_type') == 'input_text'
				and _element_hash(current) is not None
				and _element_hash(current) == _element_hash(nxt)
			):
				_discard(
					current,
					'merged_click_into_following_input_same_element',
					discarded,
				)
				after_rule3.append(nxt)
				i += 2
				continue
			if (
				nxt is not None
				and current.get('action_type') == 'input_text'
				and nxt.get('action_type') == 'key_press'
				and str(nxt.get('typed_value') or '').lower() in {'enter', 'return'}
			):
				merged = dict(current)
				merged['press_enter'] = True
				# Prefer navigation signal from either half of the merge.
				merged['caused_navigation'] = bool(current.get('caused_navigation')) or bool(
					nxt.get('caused_navigation')
				)
				if nxt.get('next_page_url'):
					merged['next_page_url'] = nxt.get('next_page_url')
				_discard(nxt, 'merged_enter_keypress_into_preceding_input', discarded)
				logger.info(
					f'kept step_index={merged.get("step_index")} action_type=input_text: '
					f'merged_with_following_enter_keypress'
				)
				after_rule3.append(merged)
				i += 2
				continue
			after_rule3.append(current)
			i += 1

		# Rule 4: if the same element is acted on multiple times, keep only the
		# LAST successful action on it. Records without an element_hash (e.g.
		# navigate) are not keyed and always survive this pass.
		last_index_by_hash: dict[str, int] = {}
		for idx, record in enumerate(after_rule3):
			eh = _element_hash(record)
			if eh is not None:
				last_index_by_hash[eh] = idx

		# Rule 5: keep everything remaining that mutates page state.
		kept: list[dict[str, Any]] = []
		for idx, record in enumerate(after_rule3):
			eh = _element_hash(record)
			if eh is not None and last_index_by_hash.get(eh) != idx:
				_discard(record, 'superseded_by_later_action_on_same_element', discarded)
				continue
			if eh is not None:
				_keep(record, 'last_successful_action_on_element', kept)
			else:
				_keep(record, f'mutating_action:{record.get("action_type")}', kept)

		reason_counts = Counter(entry['reason'] for entry in discarded)
		logger.info(
			f'filter_steps summary: total={len(records)} kept={len(kept)} '
			f'discarded={len(discarded)} reasons={dict(reason_counts)}'
		)
		return kept, discarded


def filter_steps(
	records: list[dict[str, Any]],
	step_filter: StepFilter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	"""Filter exported records down to replayable steps.

	Defaults to `RuleBasedStepFilter`. Pass any `StepFilter` implementation to
	swap in an alternate strategy (e.g. an LLM-based filter) later.
	"""
	impl: StepFilter = step_filter if step_filter is not None else RuleBasedStepFilter()
	return impl.filter_steps(records)
