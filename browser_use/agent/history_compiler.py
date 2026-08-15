from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = '1.0'
MAX_LOCATOR_VALUE_LENGTH = 512
MAX_XPATH_LENGTH = 4096

_HTML_TAG_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9-]*$')
_DYNAMIC_PREFIX_PATTERN = re.compile(r'^(?:ember|react|vue|ng|mui|chakra|radix|headlessui)[-_]?\d', re.IGNORECASE)
_UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.IGNORECASE)
_CONTROL_CHARACTER_PATTERN = re.compile(r'[\x00-\x1f\x7f]')
_SECRET_PLACEHOLDER_PATTERN = re.compile(r'<secret>[^<>]+</secret>')

_LOCATOR_ATTRIBUTE_PRIORITY: tuple[tuple[str, int], ...] = (
	('data-testid', 100),
	('data-test-id', 98),
	('data-test', 98),
	('data-cy', 98),
	('data-qa', 98),
	('id', 92),
	('name', 84),
)
_LOCATOR_RELEVANT_ATTRIBUTES = frozenset(
	{
		'data-testid',
		'data-test-id',
		'data-test',
		'data-cy',
		'data-qa',
		'id',
		'name',
		'aria-label',
		'placeholder',
		'role',
		'type',
	}
)


class HistoryCompilationError(ValueError):
	"""Raised when a Browser Use history file cannot be compiled safely."""


class CacheModel(BaseModel):
	"""Base model for the versioned Browser Use action-cache contract."""

	model_config = ConfigDict(extra='forbid')


class BrowserActionExecutionStatus(str, Enum):
	EXECUTED_WITHOUT_REPORTED_ERROR = 'executed_without_reported_error'
	FAILED = 'failed'
	NOT_EXECUTED = 'not_executed'
	RESULT_UNKNOWN = 'result_unknown'
	TASK_COMPLETED_SUCCESSFULLY = 'task_completed_successfully'
	TASK_COMPLETED_UNSUCCESSFULLY = 'task_completed_unsuccessfully'


class CachedAutomationDecisionStatus(str, Enum):
	WAITING_FOR_LOCATOR_VALIDATION = 'waiting_for_locator_validation'
	REQUIRES_AGENTIC_HANDLING = 'requires_agentic_handling'
	MANUAL_REVIEW_REQUIRED = 'manual_review_required'
	EXCLUDE_TERMINAL_ACTION = 'exclude_terminal_action'
	EXCLUDE_FAILED_ACTION = 'exclude_failed_action'
	EXCLUDE_NOT_EXECUTED_ACTION = 'exclude_not_executed_action'
	WITHHOLD_SOURCE_RUN_NOT_SUCCESSFUL = 'withhold_source_run_not_successful'


class RedundancyStatus(str, Enum):
	NO_REDUNDANCY_DETECTED = 'no_redundancy_detected'
	SUSPECTED_REDUNDANT = 'suspected_redundant'
	CONFIRMED_REDUNDANT = 'confirmed_redundant'
	CONFIRMED_REQUIRED = 'confirmed_required'


class CacheStatus(str, Enum):
	DRAFT_REQUIRES_LOCATOR_VALIDATION = 'draft_requires_locator_validation'
	DRAFT_REQUIRES_AGENTIC_HANDLING = 'draft_requires_agentic_handling'
	NEEDS_MANUAL_REVIEW = 'needs_manual_review'
	SOURCE_RUN_NOT_SUCCESSFUL = 'source_run_not_successful'
	NO_DETERMINISTIC_CANDIDATES = 'no_deterministic_candidates'


class HistoryLocation(CacheModel):
	browser_use_history_item: int = Field(ge=0)
	action_inside_history_item: int = Field(ge=0)


class CompilationIssue(CacheModel):
	issue_code: str
	explanation: str
	history_location: HistoryLocation | None = None


class SourceHistory(CacheModel):
	file_name: str


class OriginalRun(CacheModel):
	starting_url: str | None = None
	task_instruction: str | None = None
	task_completed: bool = False
	task_succeeded: bool | None = None


class PageBeforeActionBatch(CacheModel):
	url: str | None = None
	title: str | None = None
	snapshot_scope: Literal['history_item_before_action_batch'] = 'history_item_before_action_batch'


class BrowserAction(CacheModel):
	original_action_name: str
	action_type: str
	action_details: dict[str, Any] = Field(default_factory=dict)
	temporary_browser_use_element_index: int | None = Field(default=None, ge=0)


class BrowserActionResult(CacheModel):
	status: BrowserActionExecutionStatus
	has_reported_error: bool = False
	reported_success: bool | None = None


class ElementUsed(CacheModel):
	html_tag: str | None = None
	locator_relevant_attributes: dict[str, str] = Field(default_factory=dict)
	accessibility_name: str | None = None
	recorded_xpath: str | None = None


class LocatorValidation(CacheModel):
	status: Literal['not_tested'] = 'not_tested'


class PlaywrightLocatorOption(CacheModel):
	locator_name: str
	playwright_command: str
	built_from: list[str]
	ranking_score: int = Field(ge=0, le=100)
	appears_dynamic: bool = False
	validation: LocatorValidation = Field(default_factory=LocatorValidation)


class CachedAutomationDecision(CacheModel):
	decision: CachedAutomationDecisionStatus
	included_in_deterministic_candidates: bool
	deterministic_candidate_number: int | None = Field(default=None, ge=1)
	explanation: str

	@model_validator(mode='after')
	def validate_candidate_reference(self) -> 'CachedAutomationDecision':
		if self.included_in_deterministic_candidates != (self.deterministic_candidate_number is not None):
			raise ValueError('included_in_deterministic_candidates must match whether deterministic_candidate_number is present')
		return self


class RedundancyCheck(CacheModel):
	status: RedundancyStatus = RedundancyStatus.NO_REDUNDANCY_DETECTED
	explanation: str
	related_step_number: int | None = Field(default=None, ge=1)


class ObservedStep(CacheModel):
	step_number: int = Field(ge=1)
	history_location: HistoryLocation
	page_before_action_batch: PageBeforeActionBatch
	browser_action: BrowserAction
	browser_action_result: BrowserActionResult
	element_used: ElementUsed | None = None
	cached_automation_decision: CachedAutomationDecision
	redundancy_check: RedundancyCheck


class PlaywrightAction(CacheModel):
	action_type: Literal['fill', 'type']
	input_text: str


class DeterministicStepCandidate(CacheModel):
	candidate_number: int = Field(ge=1)
	source_step_number: int = Field(ge=1)
	playwright_action: PlaywrightAction
	playwright_locator_options: list[PlaywrightLocatorOption]
	highest_ranked_unvalidated_locator: str
	chosen_playwright_locator: str | None = None
	playwright_code_preview: str

	@model_validator(mode='after')
	def validate_locator_options(self) -> 'DeterministicStepCandidate':
		if not self.playwright_locator_options:
			raise ValueError('A deterministic step candidate requires at least one locator option')
		if self.highest_ranked_unvalidated_locator != self.playwright_locator_options[0].playwright_command:
			raise ValueError('highest_ranked_unvalidated_locator must be the first ranked locator option')
		available_commands = {option.playwright_command for option in self.playwright_locator_options}
		if self.chosen_playwright_locator is not None and self.chosen_playwright_locator not in available_commands:
			raise ValueError('chosen_playwright_locator must reference a generated locator option')
		return self


class ConversionSummary(CacheModel):
	total_observed_steps: int = Field(ge=0)
	deterministic_candidates: int = Field(ge=0)
	requires_agentic_handling: int = Field(ge=0)
	manual_review_steps: int = Field(ge=0)
	excluded_terminal_steps: int = Field(ge=0)
	excluded_failed_steps: int = Field(ge=0)
	excluded_not_executed_steps: int = Field(ge=0)
	withheld_source_run_steps: int = Field(ge=0)
	suspected_redundant_steps: int = Field(ge=0)
	issues_found: int = Field(ge=0)

	@model_validator(mode='after')
	def validate_step_count(self) -> 'ConversionSummary':
		classified_steps = (
			self.deterministic_candidates
			+ self.requires_agentic_handling
			+ self.manual_review_steps
			+ self.excluded_terminal_steps
			+ self.excluded_failed_steps
			+ self.excluded_not_executed_steps
			+ self.withheld_source_run_steps
		)
		if classified_steps != self.total_observed_steps:
			raise ValueError('Every observed step must have exactly one cached-automation decision')
		return self


class BrowserUseActionCache(CacheModel):
	cache_format_version: Literal['1.0'] = CACHE_FORMAT_VERSION
	cache_status: CacheStatus
	source_history: SourceHistory
	original_run: OriginalRun
	issues: list[CompilationIssue]
	all_observed_steps: list[ObservedStep]
	deterministic_step_candidates: list[DeterministicStepCandidate]
	conversion_summary: ConversionSummary

	@model_validator(mode='after')
	def validate_cross_references(self) -> 'BrowserUseActionCache':
		step_numbers = [step.step_number for step in self.all_observed_steps]
		if step_numbers != list(range(1, len(step_numbers) + 1)):
			raise ValueError('all_observed_steps must use consecutive step numbers starting at 1')

		candidate_numbers = [candidate.candidate_number for candidate in self.deterministic_step_candidates]
		if candidate_numbers != list(range(1, len(candidate_numbers) + 1)):
			raise ValueError('deterministic_step_candidates must use consecutive candidate numbers starting at 1')

		candidates_by_number = {candidate.candidate_number: candidate for candidate in self.deterministic_step_candidates}
		steps_by_number = {step.step_number: step for step in self.all_observed_steps}
		for step in self.all_observed_steps:
			decision = step.cached_automation_decision
			if decision.deterministic_candidate_number is None:
				continue
			candidate = candidates_by_number.get(decision.deterministic_candidate_number)
			if candidate is None or candidate.source_step_number != step.step_number:
				raise ValueError('A cached-automation decision references an unknown deterministic candidate')
		for candidate in self.deterministic_step_candidates:
			source_step = steps_by_number.get(candidate.source_step_number)
			if (
				source_step is None
				or source_step.cached_automation_decision.deterministic_candidate_number != candidate.candidate_number
			):
				raise ValueError('A deterministic candidate references an unknown source step')

		if self.conversion_summary != _build_summary(self.all_observed_steps, self.issues):
			raise ValueError('conversion_summary does not match the cache contents')
		return self


def compile_history_to_action_cache(
	history_path: str | Path,
	cache_path: str | Path,
	*,
	task_instruction: str | None = None,
) -> BrowserUseActionCache:
	"""Compile a saved Browser Use history into an auditable Playwright action cache.

	The compiler is deliberately offline and deterministic: it does not call an LLM,
	does not access a live browser, and does not claim that locator candidates are
	validated. Every proposed Browser Use action is retained in ``all_observed_steps``
	with an explicit decision explaining whether it can contribute to deterministic
	replay.

	Args:
		history_path: Path to a Browser Use ``raw_history.json`` file.
		cache_path: Destination for the compiled action-cache JSON.
		task_instruction: Optional original task text for provenance.

	Returns:
		The validated cache model that was written to ``cache_path``.

	Raises:
		HistoryCompilationError: If the history file or its root structure is invalid.
		OSError: If the cache cannot be written atomically.
	"""
	history_file = Path(history_path)
	cache_file = Path(cache_path)
	if history_file.resolve() == cache_file.resolve():
		raise HistoryCompilationError('History and cache paths must be different files')

	try:
		with history_file.open(encoding='utf-8') as file:
			raw_history = json.load(file)
	except (OSError, json.JSONDecodeError) as exc:
		raise HistoryCompilationError(f'Could not read Browser Use history file {history_file.name!r}') from exc

	cache = compile_history_data(
		raw_history,
		source_file_name=history_file.name,
		task_instruction=task_instruction,
	)
	_write_cache_atomically(cache, cache_file)

	summary = cache.conversion_summary
	logger.info(
		'Compiled Browser Use action cache: observed=%d candidates=%d agentic=%d manual_review=%d '
		'excluded_terminal=%d excluded_failed=%d excluded_not_executed=%d withheld_source_run=%d '
		'redundancy_candidates=%d issues=%d path=%s',
		summary.total_observed_steps,
		summary.deterministic_candidates,
		summary.requires_agentic_handling,
		summary.manual_review_steps,
		summary.excluded_terminal_steps,
		summary.excluded_failed_steps,
		summary.excluded_not_executed_steps,
		summary.withheld_source_run_steps,
		summary.suspected_redundant_steps,
		summary.issues_found,
		cache_file,
	)
	return cache


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
	deterministic_candidates: list[DeterministicStepCandidate] = []
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
			action_name, action_arguments, action_shape_valid = _parse_action(raw_action, location, issues)

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
			element_used = _normalise_element(raw_element, location, issues)
			browser_action = _normalise_browser_action(action_name, action_arguments)

			decision, candidate = _compile_action(
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
					run_succeeded = raw_result.get('success') if isinstance(raw_result.get('success'), bool) else None
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
			if decision.decision != CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION:
				continue
			observed_step.cached_automation_decision = CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.WITHHOLD_SOURCE_RUN_NOT_SUCCESSFUL,
				included_in_deterministic_candidates=False,
				explanation=(
					'The step has replay evidence, but the overall source run did not finish successfully, '
					'so it cannot be promoted to deterministic replay.'
				),
			)

	summary = _build_summary(observed_steps, issues)
	cache_status = _determine_cache_status(run_completed, run_succeeded, summary)

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


def _parse_action(
	raw_action: Any,
	location: HistoryLocation,
	issues: list[CompilationIssue],
) -> tuple[str, dict[str, Any], bool]:
	if not isinstance(raw_action, Mapping):
		issues.append(
			CompilationIssue(
				issue_code='ACTION_NOT_OBJECT',
				explanation='The proposed action is not a JSON object.',
				history_location=location,
			)
		)
		return 'invalid_action', {}, False

	action_entries = [(key, value) for key, value in raw_action.items() if value is not None]
	if len(action_entries) != 1 or not isinstance(action_entries[0][0], str):
		issues.append(
			CompilationIssue(
				issue_code='ACTION_NOT_SINGLE_KEY',
				explanation='Each proposed action must contain exactly one non-null action type.',
				history_location=location,
			)
		)
		return 'invalid_action', {}, False

	action_name, raw_arguments = action_entries[0]
	if not isinstance(raw_arguments, Mapping):
		issues.append(
			CompilationIssue(
				issue_code='ACTION_ARGUMENTS_NOT_OBJECT',
				explanation='The proposed action arguments are not a JSON object.',
				history_location=location,
			)
		)
		return action_name, {}, False

	return action_name, dict(raw_arguments), True


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
	if action_name == 'done' and raw_result.get('is_done') is True:
		return BrowserActionResult(
			status=(
				BrowserActionExecutionStatus.TASK_COMPLETED_SUCCESSFULLY
				if reported_success is True and not has_error
				else BrowserActionExecutionStatus.TASK_COMPLETED_UNSUCCESSFULLY
			),
			has_reported_error=has_error,
			reported_success=reported_success,
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


def _normalise_browser_action(action_name: str, action_arguments: Mapping[str, Any]) -> BrowserAction:
	readable_action_names = {
		'input': 'enter_text',
		'click': 'click_element',
		'navigate': 'go_to_url',
		'select_dropdown': 'select_dropdown_option',
		'send_keys': 'press_keys',
		'done': 'report_task_completion',
	}
	action_details = {key: value for key, value in action_arguments.items() if key != 'index'}
	if action_name == 'input':
		input_text = action_arguments.get('text')
		secret_references = _SECRET_PLACEHOLDER_PATTERN.findall(input_text) if isinstance(input_text, str) else []
		action_details = {'clear_existing_text': action_arguments.get('clear', True)}
		if secret_references:
			action_details['sensitive_parameter_references'] = secret_references
		else:
			action_details['text'] = input_text
	elif action_name == 'navigate':
		action_details = {
			'url': action_arguments.get('url'),
			'open_in_new_tab': action_arguments.get('new_tab', False),
		}
	elif action_name == 'done':
		# The raw history remains the source of truth for the model's completion
		# message. Avoid copying that potentially sensitive text into the cache.
		action_details = {}

	index = action_arguments.get('index')
	temporary_element_index = index if isinstance(index, int) and not isinstance(index, bool) and index >= 0 else None
	return BrowserAction(
		original_action_name=action_name,
		action_type=readable_action_names.get(action_name, action_name),
		action_details=action_details,
		temporary_browser_use_element_index=temporary_element_index,
	)


def _normalise_element(
	raw_element: Any,
	location: HistoryLocation,
	issues: list[CompilationIssue],
) -> ElementUsed | None:
	if raw_element is None:
		return None
	if not isinstance(raw_element, Mapping):
		issues.append(
			CompilationIssue(
				issue_code='INTERACTED_ELEMENT_NOT_OBJECT',
				explanation='The interacted element is not a JSON object.',
				history_location=location,
			)
		)
		return None

	node_name = raw_element.get('node_name')
	html_tag = node_name.lower() if isinstance(node_name, str) and _HTML_TAG_PATTERN.fullmatch(node_name) else None

	raw_attributes = raw_element.get('attributes')
	attributes: dict[str, str] = {}
	if isinstance(raw_attributes, Mapping):
		for key in sorted(_LOCATOR_RELEVANT_ATTRIBUTES):
			value = raw_attributes.get(key)
			if isinstance(value, str):
				attributes[key] = value
	elif raw_attributes is not None:
		issues.append(
			CompilationIssue(
				issue_code='ELEMENT_ATTRIBUTES_NOT_OBJECT',
				explanation='The interacted element attributes are not a JSON object.',
				history_location=location,
			)
		)

	ax_name = _safe_locator_value(raw_element.get('ax_name'))
	xpath = _safe_xpath(raw_element.get('x_path'))
	return ElementUsed(
		html_tag=html_tag,
		locator_relevant_attributes=attributes,
		accessibility_name=ax_name,
		recorded_xpath=xpath,
	)


def _compile_action(
	*,
	step_number: int,
	action_name: str,
	action_arguments: Mapping[str, Any],
	action_shape_valid: bool,
	execution_result: BrowserActionResult,
	element_used: ElementUsed | None,
	location: HistoryLocation,
	issues: list[CompilationIssue],
	candidate_number: int,
) -> tuple[CachedAutomationDecision, DeterministicStepCandidate | None]:
	if action_name == 'done' and execution_result.status in {
		BrowserActionExecutionStatus.TASK_COMPLETED_SUCCESSFULLY,
		BrowserActionExecutionStatus.TASK_COMPLETED_UNSUCCESSFULLY,
	}:
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION,
				included_in_deterministic_candidates=False,
				explanation='This action reports task completion and does not interact with the webpage.',
			),
			None,
		)

	if execution_result.status == BrowserActionExecutionStatus.FAILED:
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.EXCLUDE_FAILED_ACTION,
				included_in_deterministic_candidates=False,
				explanation='The recorded action reported a failure and cannot be promoted to deterministic replay.',
			),
			None,
		)

	if execution_result.status == BrowserActionExecutionStatus.NOT_EXECUTED:
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.EXCLUDE_NOT_EXECUTED_ACTION,
				included_in_deterministic_candidates=False,
				explanation='Browser Use stopped the action batch before this proposed action was executed.',
			),
			None,
		)

	if not action_shape_valid or execution_result.status == BrowserActionExecutionStatus.RESULT_UNKNOWN:
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.MANUAL_REVIEW_REQUIRED,
				included_in_deterministic_candidates=False,
				explanation='The action or its result is malformed or incomplete and requires manual review.',
			),
			None,
		)

	if action_name != 'input':
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING,
				included_in_deterministic_candidates=False,
				explanation=(
					'This action type is preserved in the audit but is not compiled by the current input-action adapter.'
				),
			),
			None,
		)

	input_text = action_arguments.get('text')
	clear_existing_text = action_arguments.get('clear', True)
	temporary_index = action_arguments.get('index')
	if not isinstance(input_text, str) or not isinstance(clear_existing_text, bool) or not _is_int(temporary_index):
		issues.append(
			CompilationIssue(
				issue_code='INVALID_INPUT_ACTION_ARGUMENTS',
				explanation='An input action requires an integer index, string text, and boolean clear flag.',
				history_location=location,
			)
		)
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.MANUAL_REVIEW_REQUIRED,
				included_in_deterministic_candidates=False,
				explanation='The input action arguments do not match the Browser Use input-action contract.',
			),
			None,
		)

	if _SECRET_PLACEHOLDER_PATTERN.search(input_text):
		issues.append(
			CompilationIssue(
				issue_code='SENSITIVE_INPUT_REQUIRES_PARAMETER_MAPPING',
				explanation=(
					'The input contains a Browser Use secret placeholder and requires an explicit runtime parameter mapping.'
				),
				history_location=location,
			)
		)
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.MANUAL_REVIEW_REQUIRED,
				included_in_deterministic_candidates=False,
				explanation='A redacted secret must never be compiled as literal text.',
			),
			None,
		)

	if element_used is None:
		issues.append(
			CompilationIssue(
				issue_code='MISSING_REQUIRED_TARGET_EVIDENCE',
				explanation='The input action has no interacted-element evidence for generating a locator.',
				history_location=location,
			)
		)
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING,
				included_in_deterministic_candidates=False,
				explanation='No recorded element evidence is available for a deterministic input locator.',
			),
			None,
		)

	locator_options = _build_locator_options(element_used)
	if not locator_options:
		issues.append(
			CompilationIssue(
				issue_code='NO_LOCATOR_CANDIDATE',
				explanation='Recorded element evidence did not produce a safe Playwright locator candidate.',
				history_location=location,
			)
		)
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING,
				included_in_deterministic_candidates=False,
				explanation='The recorded element evidence cannot be converted into a safe Playwright locator.',
			),
			None,
		)

	playwright_action_type: Literal['fill', 'type'] = 'fill' if clear_existing_text else 'type'
	highest_ranked_locator = locator_options[0].playwright_command
	preview = f'page.{highest_ranked_locator}.{playwright_action_type}({_python_string_literal(input_text)})'
	candidate = DeterministicStepCandidate(
		candidate_number=candidate_number,
		source_step_number=step_number,
		playwright_action=PlaywrightAction(
			action_type=playwright_action_type,
			input_text=input_text,
		),
		playwright_locator_options=locator_options,
		highest_ranked_unvalidated_locator=highest_ranked_locator,
		chosen_playwright_locator=None,
		playwright_code_preview=preview,
	)
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION,
			included_in_deterministic_candidates=True,
			deterministic_candidate_number=candidate_number,
			explanation=(
				'The input action executed without a reported error and has evidence-derived locator candidates. '
				'A fresh Playwright replay must validate the chosen locator.'
			),
		),
		candidate,
	)


def _build_locator_options(element: ElementUsed) -> list[PlaywrightLocatorOption]:
	if element.html_tag is None:
		return []

	attributes = element.locator_relevant_attributes
	candidates: list[PlaywrightLocatorOption] = []
	for attribute_name, score in _LOCATOR_ATTRIBUTE_PRIORITY:
		value = _safe_locator_value(attributes.get(attribute_name))
		if value is None:
			continue
		appears_dynamic = _looks_dynamic(value)
		if attribute_name == 'data-testid':
			command = f'get_by_test_id({_python_string_literal(value)})'
		else:
			selector = _css_attribute_selector(element.html_tag, attribute_name, value)
			command = f'locator({_python_string_literal(selector)})'
		candidates.append(
			PlaywrightLocatorOption(
				locator_name=_readable_locator_name(attribute_name),
				playwright_command=command,
				built_from=['element_used.html_tag', f'element_used.locator_relevant_attributes.{attribute_name}'],
				ranking_score=_adjust_locator_score(score, appears_dynamic),
				appears_dynamic=appears_dynamic,
			)
		)

	name = _safe_locator_value(attributes.get('name'))
	input_type = _safe_locator_value(attributes.get('type'))
	if name and input_type:
		appears_dynamic = _looks_dynamic(name) or _looks_dynamic(input_type)
		selector = (
			f'{element.html_tag}[name="{_escape_css_attribute_value(name)}"][type="{_escape_css_attribute_value(input_type)}"]'
		)
		candidates.append(
			PlaywrightLocatorOption(
				locator_name='name_and_type_attributes',
				playwright_command=f'locator({_python_string_literal(selector)})',
				built_from=[
					'element_used.html_tag',
					'element_used.locator_relevant_attributes.name',
					'element_used.locator_relevant_attributes.type',
				],
				ranking_score=_adjust_locator_score(82, appears_dynamic),
				appears_dynamic=appears_dynamic,
			)
		)

	aria_label = _safe_locator_value(attributes.get('aria-label'))
	if aria_label:
		appears_dynamic = _looks_dynamic(aria_label)
		candidates.append(
			PlaywrightLocatorOption(
				locator_name='aria_label',
				playwright_command=f'get_by_label({_python_string_literal(aria_label)})',
				built_from=['element_used.locator_relevant_attributes.aria-label'],
				ranking_score=_adjust_locator_score(76, appears_dynamic),
				appears_dynamic=appears_dynamic,
			)
		)

	role = _safe_locator_value(attributes.get('role'))
	accessible_name = _safe_locator_value(element.accessibility_name)
	if role and accessible_name:
		appears_dynamic = _looks_dynamic(role) or _looks_dynamic(accessible_name)
		candidates.append(
			PlaywrightLocatorOption(
				locator_name='role_and_accessible_name',
				playwright_command=(
					f'get_by_role({_python_string_literal(role)}, name={_python_string_literal(accessible_name)})'
				),
				built_from=[
					'element_used.locator_relevant_attributes.role',
					'element_used.accessibility_name',
				],
				ranking_score=_adjust_locator_score(72, appears_dynamic),
				appears_dynamic=appears_dynamic,
			)
		)

	placeholder = _safe_locator_value(attributes.get('placeholder'))
	if placeholder:
		appears_dynamic = _looks_dynamic(placeholder)
		candidates.append(
			PlaywrightLocatorOption(
				locator_name='placeholder',
				playwright_command=f'get_by_placeholder({_python_string_literal(placeholder)})',
				built_from=['element_used.locator_relevant_attributes.placeholder'],
				ranking_score=_adjust_locator_score(64, appears_dynamic),
				appears_dynamic=appears_dynamic,
			)
		)

	if element.recorded_xpath:
		xpath = element.recorded_xpath if element.recorded_xpath.startswith('/') else f'/{element.recorded_xpath}'
		candidates.append(
			PlaywrightLocatorOption(
				locator_name='recorded_xpath',
				playwright_command=f'locator({_python_string_literal(f"xpath={xpath}")})',
				built_from=['element_used.recorded_xpath'],
				ranking_score=10,
			)
		)

	unique_candidates: dict[str, PlaywrightLocatorOption] = {}
	for candidate in candidates:
		unique_candidates.setdefault(candidate.playwright_command, candidate)
	return sorted(
		unique_candidates.values(),
		key=lambda candidate: (-candidate.ranking_score, candidate.playwright_command),
	)


def _build_summary(observed_steps: list[ObservedStep], issues: list[CompilationIssue]) -> ConversionSummary:
	decisions = [step.cached_automation_decision.decision for step in observed_steps]
	return ConversionSummary(
		total_observed_steps=len(observed_steps),
		deterministic_candidates=decisions.count(CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION),
		requires_agentic_handling=decisions.count(CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING),
		manual_review_steps=decisions.count(CachedAutomationDecisionStatus.MANUAL_REVIEW_REQUIRED),
		excluded_terminal_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION),
		excluded_failed_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_FAILED_ACTION),
		excluded_not_executed_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_NOT_EXECUTED_ACTION),
		withheld_source_run_steps=decisions.count(CachedAutomationDecisionStatus.WITHHOLD_SOURCE_RUN_NOT_SUCCESSFUL),
		suspected_redundant_steps=sum(
			step.redundancy_check.status == RedundancyStatus.SUSPECTED_REDUNDANT for step in observed_steps
		),
		issues_found=len(issues),
	)


def _determine_cache_status(
	run_completed: bool,
	run_succeeded: bool | None,
	summary: ConversionSummary,
) -> CacheStatus:
	if not run_completed or run_succeeded is not True:
		return CacheStatus.SOURCE_RUN_NOT_SUCCESSFUL
	if summary.issues_found or summary.manual_review_steps:
		return CacheStatus.NEEDS_MANUAL_REVIEW
	if summary.requires_agentic_handling:
		return CacheStatus.DRAFT_REQUIRES_AGENTIC_HANDLING
	if summary.deterministic_candidates:
		return CacheStatus.DRAFT_REQUIRES_LOCATOR_VALIDATION
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


def _write_cache_atomically(cache: BrowserUseActionCache, cache_path: Path) -> None:
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	serialised_cache = (
		json.dumps(
			cache.model_dump(mode='json'),
			indent=2,
			sort_keys=True,
			ensure_ascii=False,
			allow_nan=False,
		)
		+ '\n'
	)
	temporary_path: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			mode='w',
			encoding='utf-8',
			dir=cache_path.parent,
			prefix=f'.{cache_path.name}.',
			suffix='.tmp',
			delete=False,
		) as temporary_file:
			temporary_path = Path(temporary_file.name)
			os.chmod(temporary_path, 0o600)
			temporary_file.write(serialised_cache)
			temporary_file.flush()
			os.fsync(temporary_file.fileno())
		os.replace(temporary_path, cache_path)
	except Exception:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)
		raise


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


def _safe_locator_value(value: Any) -> str | None:
	if not isinstance(value, str):
		return None
	if not value.strip() or len(value) > MAX_LOCATOR_VALUE_LENGTH or _CONTROL_CHARACTER_PATTERN.search(value):
		return None
	return value


def _safe_xpath(value: Any) -> str | None:
	if not isinstance(value, str):
		return None
	value = value.strip()
	if not value or len(value) > MAX_XPATH_LENGTH or _CONTROL_CHARACTER_PATTERN.search(value):
		return None
	return value


def _css_attribute_selector(tag: str, attribute_name: str, value: str) -> str:
	return f'{tag}[{attribute_name}="{_escape_css_attribute_value(value)}"]'


def _escape_css_attribute_value(value: str) -> str:
	return value.replace('\\', '\\\\').replace('"', '\\"')


def _python_string_literal(value: str) -> str:
	return json.dumps(value, ensure_ascii=False)


def _readable_locator_name(attribute_name: str) -> str:
	return {
		'data-testid': 'test_id',
		'data-test-id': 'data_test_id_attribute',
		'data-test': 'data_test_attribute',
		'data-cy': 'data_cy_attribute',
		'data-qa': 'data_qa_attribute',
		'id': 'id_attribute',
		'name': 'name_attribute',
	}.get(attribute_name, f'{attribute_name}_attribute')


def _adjust_locator_score(base_score: int, appears_dynamic: bool) -> int:
	"""Keep evidence-derived locators while ranking likely generated values lower."""
	return max(1, base_score - 40) if appears_dynamic else base_score


def _looks_dynamic(value: str) -> bool:
	if _UUID_PATTERN.search(value) or _DYNAMIC_PREFIX_PATTERN.match(value):
		return True
	if re.search(r'\d{4,}', value):
		return True
	for segment in re.split(r'[\s_-]+', value):
		if len(segment) < 5:
			continue
		digit_count = sum(character.isdigit() for character in segment)
		has_alpha = any(character.isalpha() for character in segment)
		vowel_count = sum(character in 'aeiouAEIOU' for character in segment)
		# Generated IDs commonly mix several digits with a low-vowel random token.
		# Do not reject meaningful names with numeric prefixes (for example,
		# ``04fullname`` or ``10address1``), which are stable on some forms.
		if has_alpha and len(segment) >= 8 and digit_count >= 3 and vowel_count <= 1:
			return True
		if has_alpha and vowel_count == 0 and len(segment) >= 6:
			return True
	return False


def _optional_string(value: Any) -> str | None:
	return value if isinstance(value, str) else None


def _non_empty_string(value: Any) -> bool:
	return isinstance(value, str) and bool(value.strip())


def _is_int(value: Any) -> bool:
	return isinstance(value, int) and not isinstance(value, bool)
