from __future__ import annotations

from copy import deepcopy

from browser_use.agent.history_compiler import (
	BrowserActionExecutionStatus,
	CachedAutomationDecisionStatus,
	DeterministicStepCandidate,
	compile_history_data,
)


def _successful_input_history() -> dict:
	return {
		'history': [
			{
				'state': {
					'url': 'https://example.test/form',
					'title': 'Form',
					'interacted_element': [
						{
							'node_name': 'INPUT',
							'attributes': {'id': 'display-name', 'type': 'text'},
							'ax_name': 'Display name',
							'x_path': 'html/body/form/input',
						}
					],
				},
				'model_output': {'action': [{'input': {'index': 17, 'text': 'Ada', 'clear': True}}]},
				'result': [{'success': True}],
			},
			{
				'state': {
					'url': 'https://example.test/form',
					'title': 'Form',
					'interacted_element': [None],
				},
				'model_output': {'action': [{'done': {'text': 'Complete'}}]},
				'result': [
					{
						'is_done': True,
						'success': True,
						'judgement': {'verdict': True},
					}
				],
			},
		]
	}


def test_compiler_preserves_order_and_uses_recorded_element_evidence() -> None:
	cache = compile_history_data(_successful_input_history())

	assert [step.browser_action.original_action_name for step in cache.all_observed_steps] == ['input', 'done']
	assert cache.original_run.task_succeeded is True
	assert len(cache.deterministic_step_candidates) == 1
	candidate = cache.deterministic_step_candidates[0]
	assert isinstance(candidate, DeterministicStepCandidate)
	assert candidate.source_step_number == 1
	assert candidate.highest_ranked_unvalidated_locator == 'locator("input[id=\\"display-name\\"]")'
	assert '17' not in candidate.highest_ranked_unvalidated_locator
	assert (
		cache.all_observed_steps[-1].cached_automation_decision.decision == CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION
	)


def test_explicit_judge_failure_withholds_replay_candidates() -> None:
	history = deepcopy(_successful_input_history())
	history['history'][-1]['result'][0]['judgement'] = {
		'verdict': False,
		'failure_reason': 'Expected state was not reached',
	}

	cache = compile_history_data(history)

	assert cache.original_run.task_succeeded is False
	assert cache.deterministic_step_candidates == []
	assert cache.all_observed_steps[-1].browser_action_result.status == BrowserActionExecutionStatus.TASK_COMPLETED_UNSUCCESSFULLY
	assert all(
		step.cached_automation_decision.decision == CachedAutomationDecisionStatus.WITHHOLD_SOURCE_RUN_NOT_SUCCESSFUL
		for step in cache.all_observed_steps[:-1]
	)
