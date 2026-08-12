"""Unit tests for browser_use.cache.step_filter (Phase 2 / T2)."""

from __future__ import annotations

from typing import Any

from browser_use.cache.step_filter import RuleBasedStepFilter, filter_steps


def _elem(element_hash: str, name: str = 'field') -> dict[str, Any]:
	return {
		'tag': 'input',
		'id': None,
		'name': name,
		'placeholder': None,
		'aria_label': None,
		'role': None,
		'visible_text': None,
		'data_attrs': {},
		'x_path': f'html/body/input[@name="{name}"]',
		'in_shadow_dom': False,
		'element_hash': element_hash,
	}


def _rec(
	*,
	step_index: int,
	action_type: str,
	success: bool = True,
	element_hash: str | None = None,
	name: str = 'field',
	typed_value: str | None = None,
) -> dict[str, Any]:
	return {
		'step_index': step_index,
		'action_type': action_type,
		'typed_value': typed_value,
		'success': success,
		'element': _elem(element_hash, name) if element_hash else None,
		'caused_navigation': False,
		'raw': {'action_key': action_type},
	}


class TestRuleBasedStepFilter:
	"""Synthetic record list covering rules 1-4 (T2)."""

	def test_drops_failed_click_then_type_merge_last_wins_and_extraction(self):
		# One failed action, one click-then-type on same hash, one wrong-then-
		# corrected pair, one extraction step - plus a clean navigate that must
		# survive as mutating state.
		records = [
			_rec(step_index=1, action_type='other', typed_value=None),  # extraction/done
			_rec(
				step_index=2,
				action_type='click',
				success=False,
				element_hash='hash_fail',
				name='broken',
			),
			_rec(
				step_index=3,
				action_type='click',
				element_hash='hash_focus',
				name='email',
			),
			_rec(
				step_index=3,
				action_type='input_text',
				element_hash='hash_focus',
				name='email',
				typed_value='final@example.com',
			),
			_rec(
				step_index=4,
				action_type='input_text',
				element_hash='hash_retry',
				name='city',
				typed_value='wrong',
			),
			_rec(
				step_index=5,
				action_type='input_text',
				element_hash='hash_retry',
				name='city',
				typed_value='SF',
			),
			{
				'step_index': 6,
				'action_type': 'navigate',
				'typed_value': 'https://example.com',
				'success': True,
				'element': None,
				'caused_navigation': True,
				'raw': {'action_key': 'navigate'},
			},
		]

		kept, discarded = filter_steps(records)

		discard_reasons = [d['reason'] for d in discarded]
		assert 'non_browser_action:other' in discard_reasons
		assert 'failed_action' in discard_reasons
		assert 'merged_click_into_following_input_same_element' in discard_reasons
		assert 'superseded_by_later_action_on_same_element' in discard_reasons

		assert len(kept) == 3
		assert [(k['action_type'], k.get('typed_value')) for k in kept] == [
			('input_text', 'final@example.com'),
			('input_text', 'SF'),
			('navigate', 'https://example.com'),
		]

		# Superseded record must be the wrong-value attempt, not the correction.
		superseded = [
			d['record'] for d in discarded if d['reason'] == 'superseded_by_later_action_on_same_element'
		]
		assert len(superseded) == 1
		assert superseded[0]['typed_value'] == 'wrong'

		merged = [
			d['record'] for d in discarded if d['reason'] == 'merged_click_into_following_input_same_element'
		]
		assert len(merged) == 1
		assert merged[0]['action_type'] == 'click'
		assert merged[0]['element']['element_hash'] == 'hash_focus'

	def test_protocol_swap_uses_injected_filter(self):
		class KeepNothing:
			def filter_steps(self, records):
				return [], [{'reason': 'dropped_all', 'record': r} for r in records]

		kept, discarded = filter_steps(
			[_rec(step_index=1, action_type='click', element_hash='h')],
			step_filter=KeepNothing(),
		)
		assert kept == []
		assert len(discarded) == 1
		assert discarded[0]['reason'] == 'dropped_all'

	def test_default_impl_is_rule_based(self):
		assert isinstance(RuleBasedStepFilter(), RuleBasedStepFilter)
		kept, discarded = RuleBasedStepFilter().filter_steps(
			[_rec(step_index=1, action_type='other')]
		)
		assert kept == []
		assert discarded[0]['reason'] == 'non_browser_action:other'
