from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from browser_use.agent.history_cache.action_adapters import compile_action, normalise_browser_action, parse_action
from browser_use.agent.history_cache.locators import normalise_element
from browser_use.agent.history_cache.models import (
	BrowserActionExecutionStatus,
	BrowserActionResult,
	BrowserUseActionCache,
	CachedAutomationDecision,
	CachedAutomationDecisionStatus,
	CachedStepCandidate,
	CacheStatus,
	CompilationIssue,
	ConversionSummary,
	HistoryCompilationError,
	HistoryLocation,
	ObservedStep,
	OriginalRun,
	PageBeforeActionBatch,
	RedundancyCheck,
	RedundancyStatus,
	SourceHistory,
	build_conversion_summary,
)


def compile_history_data(
	raw_history: Mapping[str, Any],
	*,
	source_file_name: str = 'raw_history.json',
	task_instruction: str | None = None,
) -> BrowserUseActionCache:
	"""Compile an already-loaded Browser Use history without file-system side effects."""
	if not isinstance(raw_history, Mapping):
		raise HistoryCompilationError('Browser Use history root must be a JSON object')

	history_items = raw_history.get('history')
	if not isinstance(history_items, list):
		raise HistoryCompilationError("Browser Use history root must contain a 'history' array")

	issues: list[CompilationIssue] = []
	observed_steps: list[ObservedStep] = []
	deterministic_candidates: list[CachedStepCandidate] = []
	starting_url: str | None = None
	run_completed = False
	run_succeeded: bool | None = None

	if not history_items:
		issues.append(
			CompilationIssue(
				issue_code='EMPTY_HISTORY',
				explanation='The Browser Use history contains no history items.',
			)
		)

	for history_item_index, raw_history_item in enumerate(history_items):
		if not isinstance(raw_history_item, Mapping):
			issues.append(
				CompilationIssue(
					issue_code='HISTORY_ITEM_NOT_OBJECT',
					explanation='A Browser Use history item is not a JSON object and could not be inspected.',
					history_location=HistoryLocation(
						browser_use_history_item=history_item_index,
						action_inside_history_item=0,
					),
				)
			)
			continue

		state = raw_history_item.get('state')
		state_mapping = state if isinstance(state, Mapping) else {}
		if state is not None and not isinstance(state, Mapping):
			_add_item_issue(issues, 'STATE_NOT_OBJECT', 'The history item state is not a JSON object.', history_item_index)

		page_url = _optional_string(state_mapping.get('url'))
		page_title = _optional_string(state_mapping.get('title'))
		if starting_url is None and page_url:
			starting_url = page_url

		model_output = raw_history_item.get('model_output')
		if model_output is None:
			_add_item_issue(
				issues,
				'MODEL_OUTPUT_MISSING',
				'The history item has no model output, so it contains no proposed actions to compile.',
				history_item_index,
			)
			continue
		if not isinstance(model_output, Mapping):
			_add_item_issue(
				issues, 'MODEL_OUTPUT_NOT_OBJECT', 'The history item model output is not a JSON object.', history_item_index
			)
			continue

		raw_actions = model_output.get('action')
		if not isinstance(raw_actions, list):
			_add_item_issue(issues, 'ACTIONS_NOT_LIST', 'The model output action field is not an array.', history_item_index)
			continue

		raw_results_value = raw_history_item.get('result')
		if isinstance(raw_results_value, list):
			raw_results = raw_results_value
		else:
			raw_results = []
			_add_item_issue(issues, 'RESULTS_NOT_LIST', 'The history item result field is not an array.', history_item_index)

		raw_elements_value = state_mapping.get('interacted_element')
		if isinstance(raw_elements_value, list):
			raw_elements = raw_elements_value
		else:
			raw_elements = []
			if raw_elements_value is not None:
				_add_item_issue(
					issues,
					'INTERACTED_ELEMENTS_NOT_LIST',
					'The interacted-element field is not an array.',
					history_item_index,
				)

		if len(raw_results) > len(raw_actions):
			_add_item_issue(
				issues,
				'EXTRA_RESULTS',
				'The history item has more results than proposed actions.',
				history_item_index,
			)
		if raw_elements and len(raw_elements) != len(raw_actions):
			_add_item_issue(
				issues,
				'ELEMENT_COUNT_MISMATCH',
				'The interacted-element count does not match the proposed-action count.',
				history_item_index,
			)

		batch_stopped = False
		for action_index, raw_action in enumerate(raw_actions):
			location = HistoryLocation(
				browser_use_history_item=history_item_index,
				action_inside_history_item=action_index,
			)
			step_number = len(observed_steps) + 1
			action_name, action_arguments, action_shape_valid = parse_action(raw_action, location, issues)

			result_present = action_index < len(raw_results)
			raw_result = raw_results[action_index] if result_present else None
			execution_result = _classify_execution_result(
				action_name=action_name,
				raw_result=raw_result,
				result_present=result_present,
				batch_stopped=batch_stopped,
				location=location,
				issues=issues,
			)

			if isinstance(raw_result, Mapping) and (
				_non_empty_string(raw_result.get('error')) or raw_result.get('is_done') is True
			):
				batch_stopped = True

			raw_element = raw_elements[action_index] if action_index < len(raw_elements) else None
			element_used = normalise_element(raw_element, location, issues)
			browser_action = normalise_browser_action(action_name, action_arguments)

			decision, candidate = compile_action(
				step_number=step_number,
				action_name=action_name,
				action_arguments=action_arguments,
				action_shape_valid=action_shape_valid,
				execution_result=execution_result,
				element_used=element_used,
				location=location,
				issues=issues,
				candidate_number=len(deterministic_candidates) + 1,
			)
			if candidate is not None:
				deterministic_candidates.append(candidate)

			observed_step = ObservedStep(
				step_number=step_number,
				history_location=location,
				page_before_action_batch=PageBeforeActionBatch(url=page_url, title=page_title),
				browser_action=browser_action,
				browser_action_result=execution_result,
				element_used=element_used,
				cached_automation_decision=decision,
				redundancy_check=RedundancyCheck(
					status=RedundancyStatus.NO_REDUNDANCY_DETECTED,
					explanation='No redundancy was proven from the recorded history.',
				),
			)

			if observed_steps and _is_exact_consecutive_repeat(observed_steps[-1], observed_step):
				observed_step.redundancy_check = RedundancyCheck(
					status=RedundancyStatus.SUSPECTED_REDUNDANT,
					explanation=(
						'This action exactly repeats the preceding executed action on the same recorded target. '
						'It must remain until a fresh replay proves it can be removed.'
					),
					related_step_number=observed_steps[-1].step_number,
				)

			observed_steps.append(observed_step)

			if action_name == 'done' and result_present and isinstance(raw_result, Mapping):
				if raw_result.get('is_done') is True:
					run_completed = True
					run_succeeded = execution_result.status == BrowserActionExecutionStatus.TASK_COMPLETED_SUCCESSFULLY
					action_success = action_arguments.get('success')
					if isinstance(action_success, bool) and run_succeeded is not None and action_success != run_succeeded:
						issues.append(
							CompilationIssue(
								issue_code='DONE_OUTCOME_MISMATCH',
								explanation='The done action and its result disagree about whether the task succeeded.',
								history_location=location,
							)
						)

	if not run_completed or run_succeeded is not True:
		deterministic_candidates = []
		for observed_step in observed_steps:
			decision = observed_step.cached_automation_decision
			if decision.decision not in {
				CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION,
				CachedAutomationDecisionStatus.READY_FOR_REPLAY_VALIDATION,
			}:
				continue
			observed_step.cached_automation_decision = CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.WITHHOLD_SOURCE_RUN_NOT_SUCCESSFUL,
				included_in_deterministic_candidates=False,
				explanation=(
					'The step has replay evidence, but the overall source run did not finish successfully, '
					'so it cannot be promoted to deterministic replay.'
				),
			)

	summary = build_conversion_summary(observed_steps, issues)
	cache_status = _determine_cache_status(run_completed, run_succeeded, summary, observed_steps)

	return BrowserUseActionCache(
		cache_status=cache_status,
		source_history=SourceHistory(file_name=Path(source_file_name).name),
		original_run=OriginalRun(
			starting_url=starting_url,
			task_instruction=task_instruction,
			task_completed=run_completed,
			task_succeeded=run_succeeded,
		),
		issues=issues,
		all_observed_steps=observed_steps,
		deterministic_step_candidates=deterministic_candidates,
		conversion_summary=summary,
	)


def _classify_execution_result(
	*,
	action_name: str,
	raw_result: Any,
	result_present: bool,
	batch_stopped: bool,
	location: HistoryLocation,
	issues: list[CompilationIssue],
) -> BrowserActionResult:
	if not result_present:
		if batch_stopped:
			return BrowserActionResult(status=BrowserActionExecutionStatus.NOT_EXECUTED)
		issues.append(
			CompilationIssue(
				issue_code='UNEXPECTED_MISSING_RESULT',
				explanation='The action has no matching result and no earlier batch-stop result proves it was skipped.',
				history_location=location,
			)
		)
		return BrowserActionResult(status=BrowserActionExecutionStatus.RESULT_UNKNOWN)

	if not isinstance(raw_result, Mapping):
		issues.append(
			CompilationIssue(
				issue_code='RESULT_NOT_OBJECT',
				explanation='The matching action result is not a JSON object.',
				history_location=location,
			)
		)
		return BrowserActionResult(status=BrowserActionExecutionStatus.RESULT_UNKNOWN)

	has_error = _non_empty_string(raw_result.get('error'))
	reported_success = raw_result.get('success') if isinstance(raw_result.get('success'), bool) else None
	judgement = raw_result.get('judgement')
	judge_verdict: bool | None = None
	judge_failure_reason: str | None = None
	if isinstance(judgement, Mapping):
		raw_verdict = judgement.get('verdict')
		judge_verdict = raw_verdict if isinstance(raw_verdict, bool) else None
		raw_failure_reason = judgement.get('failure_reason')
		if isinstance(raw_failure_reason, str) and raw_failure_reason.strip():
			judge_failure_reason = raw_failure_reason
	if action_name == 'done' and raw_result.get('is_done') is True:
		# Browser Use can report success from the acting agent while its final
		# judge rejects the run. Keep only the judge's decision and concise failure
		# reason in the cache, and never learn replay candidates from that run.
		task_succeeded = reported_success is True and not has_error and judge_verdict is not False
		return BrowserActionResult(
			status=(
				BrowserActionExecutionStatus.TASK_COMPLETED_SUCCESSFULLY
				if task_succeeded
				else BrowserActionExecutionStatus.TASK_COMPLETED_UNSUCCESSFULLY
			),
			has_reported_error=has_error,
			reported_success=reported_success,
			judge_verdict=judge_verdict,
			judge_failure_reason=judge_failure_reason,
		)

	if has_error or reported_success is False:
		return BrowserActionResult(
			status=BrowserActionExecutionStatus.FAILED,
			has_reported_error=has_error,
			reported_success=reported_success,
		)

	return BrowserActionResult(
		status=BrowserActionExecutionStatus.EXECUTED_WITHOUT_REPORTED_ERROR,
		reported_success=reported_success,
	)


def _determine_cache_status(
	run_completed: bool,
	run_succeeded: bool | None,
	summary: ConversionSummary,
	observed_steps: list[ObservedStep],
) -> CacheStatus:
	if not run_completed or run_succeeded is not True:
		return CacheStatus.SOURCE_RUN_NOT_SUCCESSFUL
	if summary.issues_found or summary.manual_review_steps:
		return CacheStatus.NEEDS_MANUAL_REVIEW
	if summary.deterministic_adapter_required_steps:
		return CacheStatus.DRAFT_REQUIRES_DETERMINISTIC_ADAPTER
	if summary.requires_agentic_handling or summary.unsupported_steps:
		return CacheStatus.DRAFT_REQUIRES_AGENTIC_HANDLING
	if summary.deterministic_candidates:
		if any(
			step.cached_automation_decision.decision == CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION
			for step in observed_steps
		):
			return CacheStatus.DRAFT_REQUIRES_LOCATOR_VALIDATION
		return CacheStatus.DRAFT_REQUIRES_REPLAY_VALIDATION
	return CacheStatus.NO_DETERMINISTIC_CANDIDATES


def _is_exact_consecutive_repeat(previous: ObservedStep, current: ObservedStep) -> bool:
	executed = BrowserActionExecutionStatus.EXECUTED_WITHOUT_REPORTED_ERROR
	if previous.browser_action_result.status != executed or current.browser_action_result.status != executed:
		return False
	if previous.page_before_action_batch.url != current.page_before_action_batch.url:
		return False
	return (
		previous.browser_action.action_type == current.browser_action.action_type
		and previous.browser_action.action_details == current.browser_action.action_details
		and previous.element_used == current.element_used
	)


def _add_item_issue(
	issues: list[CompilationIssue],
	issue_code: str,
	explanation: str,
	history_item_index: int,
) -> None:
	issues.append(
		CompilationIssue(
			issue_code=issue_code,
			explanation=explanation,
			history_location=HistoryLocation(
				browser_use_history_item=history_item_index,
				action_inside_history_item=0,
			),
		)
	)


def _optional_string(value: Any) -> str | None:
	return value if isinstance(value, str) else None


def _non_empty_string(value: Any) -> bool:
	return isinstance(value, str) and bool(value.strip())
