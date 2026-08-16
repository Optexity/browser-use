from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from browser_use.agent.history_cache.locators import build_locator_options, python_string_literal
from browser_use.agent.history_cache.models import (
	BrowserAction,
	BrowserActionExecutionStatus,
	BrowserActionResult,
	CachedAutomationDecision,
	CachedAutomationDecisionStatus,
	CachedPlaywrightAction,
	CompilationIssue,
	DeterministicStepCandidate,
	ElementUsed,
	HistoryLocation,
	PlaywrightAction,
	PlaywrightClickAction,
	PlaywrightSelectAction,
)
from browser_use.tools.views import ClickElementAction, InputTextAction, SelectDropdownOptionAction

_SECRET_PLACEHOLDER_PATTERN = re.compile(r'<secret>[^<>]+</secret>')
_SENSITIVE_ARGUMENT_NAME_PATTERN = re.compile(
	r'(?:password|passwd|secret|token|api[_-]?key|authorization|cookie|credential)',
	re.IGNORECASE,
)

_INPUT_ACTION_FIELDS = frozenset({'index', 'text', 'clear'})
_CLICK_ACTION_FIELDS = frozenset({'index', 'coordinate_x', 'coordinate_y'})
_SELECT_ACTION_FIELDS = frozenset({'index', 'text'})
_COMPILED_ACTION_FIELDS = {
	'input': _INPUT_ACTION_FIELDS,
	'click': _CLICK_ACTION_FIELDS,
	'select_dropdown': _SELECT_ACTION_FIELDS,
}


@dataclass(frozen=True, slots=True)
class _ActionCompilationContext:
	step_number: int
	action_name: str
	action_arguments: Mapping[str, Any]
	execution_result: BrowserActionResult
	element_used: ElementUsed | None
	location: HistoryLocation
	issues: list[CompilationIssue]
	candidate_number: int


_CompilationOutcome = tuple[CachedAutomationDecision, DeterministicStepCandidate | None]
_ActionCompiler = Callable[[_ActionCompilationContext], _CompilationOutcome]
_SourceActionModel = TypeVar('_SourceActionModel', bound=BaseModel)


def parse_action(
	raw_action: Any,
	location: HistoryLocation,
	issues: list[CompilationIssue],
) -> tuple[str, dict[str, Any], bool]:
	"""Parse one Browser Use action while retaining sanitized malformed input for audit."""
	if not isinstance(raw_action, Mapping):
		issues.append(
			CompilationIssue(
				issue_code='ACTION_NOT_OBJECT',
				explanation='The proposed action is not a JSON object.',
				history_location=location,
			)
		)
		return 'invalid_action', {'raw_action': _sanitise_audit_value(raw_action)}, False

	action_entries = [(key, value) for key, value in raw_action.items() if value is not None]
	if len(action_entries) != 1 or not isinstance(action_entries[0][0], str):
		issues.append(
			CompilationIssue(
				issue_code='ACTION_NOT_SINGLE_KEY',
				explanation='Each proposed action must contain exactly one non-null action type.',
				history_location=location,
			)
		)
		return 'invalid_action', {'raw_action': _sanitise_audit_value(dict(raw_action))}, False

	action_name, raw_arguments = action_entries[0]
	if not isinstance(raw_arguments, Mapping):
		issues.append(
			CompilationIssue(
				issue_code='ACTION_ARGUMENTS_NOT_OBJECT',
				explanation='The proposed action arguments are not a JSON object.',
				history_location=location,
			)
		)
		return action_name, {'raw_arguments': _sanitise_audit_value(raw_arguments)}, False

	return action_name, dict(raw_arguments), True


def normalise_browser_action(action_name: str, action_arguments: Mapping[str, Any]) -> BrowserAction:
	"""Convert Browser Use action data to the readable, sanitized cache representation."""
	readable_action_names = {
		'input': 'enter_text',
		'click': 'click_element',
		'navigate': 'go_to_url',
		'select_dropdown': 'select_dropdown_option',
		'send_keys': 'press_keys',
		'done': 'report_task_completion',
	}
	action_details = {
		key: _sanitise_audit_value(value, argument_name=key) for key, value in action_arguments.items() if key != 'index'
	}
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
	known_fields = _COMPILED_ACTION_FIELDS.get(action_name)
	unhandled_action_arguments = (
		{
			key: _sanitise_audit_value(action_arguments[key], argument_name=key)
			for key in sorted(set(action_arguments) - known_fields)
		}
		if known_fields is not None
		else {}
	)
	return BrowserAction(
		original_action_name=action_name,
		action_type=readable_action_names.get(action_name, action_name),
		action_details=action_details,
		temporary_browser_use_element_index=temporary_element_index,
		unhandled_action_arguments=unhandled_action_arguments,
	)


def compile_action(
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
	"""Dispatch one parsed action to its deterministic adapter or fallback decision."""
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

	context = _ActionCompilationContext(
		step_number=step_number,
		action_name=action_name,
		action_arguments=action_arguments,
		execution_result=execution_result,
		element_used=element_used,
		location=location,
		issues=issues,
		candidate_number=candidate_number,
	)
	action_compiler = _ACTION_COMPILERS.get(action_name)
	if action_compiler is None:
		return (
			CachedAutomationDecision(
				decision=CachedAutomationDecisionStatus.UNSUPPORTED_ACTION,
				included_in_deterministic_candidates=False,
				explanation=(
					f'No deterministic cache adapter is registered for Browser Use action {action_name!r}. '
					'The action remains in the ordered audit for explicit fallback or future adapter support.'
				),
			),
			None,
		)
	return action_compiler(context)


def _compile_input_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=InputTextAction,
		accepted_fields=_INPUT_ACTION_FIELDS,
		issue_code='INVALID_INPUT_ACTION_ARGUMENTS',
		explanation='An input action requires an integer index, string text, and boolean clear flag.',
	)
	if params is None:
		return _manual_review_outcome('The input action arguments do not match the Browser Use input-action contract.')

	if _SECRET_PLACEHOLDER_PATTERN.search(params.text):
		context.issues.append(
			CompilationIssue(
				issue_code='SENSITIVE_INPUT_REQUIRES_PARAMETER_MAPPING',
				explanation=(
					'The input contains a Browser Use secret placeholder and requires an explicit runtime parameter mapping.'
				),
				history_location=context.location,
			)
		)
		return _manual_review_outcome('A redacted secret must never be compiled as literal text.')

	playwright_action_type: Literal['fill', 'type'] = 'fill' if params.clear else 'type'
	return _build_element_action_candidate(
		context,
		PlaywrightAction(action_type=playwright_action_type, input_text=params.text),
	)


def _compile_click_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=ClickElementAction,
		accepted_fields=_CLICK_ACTION_FIELDS,
		issue_code='INVALID_CLICK_ACTION_ARGUMENTS',
		explanation='A click action requires an element index or a complete coordinate pair.',
	)
	if params is None:
		return _manual_review_outcome('The click action arguments do not match the Browser Use click-action contract.')

	if params.index is None:
		if params.coordinate_x is None or params.coordinate_y is None:
			context.issues.append(
				CompilationIssue(
					issue_code='INCOMPLETE_COORDINATE_CLICK',
					explanation='A coordinate click must contain both coordinate_x and coordinate_y.',
					history_location=context.location,
				)
			)
			return _manual_review_outcome('The coordinate click is incomplete and cannot be replayed safely.')
		return _agentic_outcome(
			'A coordinate-only click has no stable target identity and requires an agentic or manually authored fallback.'
		)

	if context.element_used is not None and context.element_used.html_tag == 'select':
		return _agentic_outcome(
			'Browser Use treats clicking a native select as dropdown inspection, so it cannot be replayed as a mutating click.'
		)

	return _build_element_action_candidate(context, PlaywrightClickAction())


def _compile_select_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=SelectDropdownOptionAction,
		accepted_fields=_SELECT_ACTION_FIELDS,
		issue_code='INVALID_SELECT_ACTION_ARGUMENTS',
		explanation='A select-dropdown action requires an integer index and string option text.',
	)
	if params is None or params.index < 0:
		if params is not None:
			context.issues.append(
				CompilationIssue(
					issue_code='INVALID_SELECT_ACTION_ARGUMENTS',
					explanation='A select-dropdown action requires a non-negative element index.',
					history_location=context.location,
				)
			)
		return _manual_review_outcome('The select-dropdown arguments do not match the Browser Use select-action contract.')

	if _SECRET_PLACEHOLDER_PATTERN.search(params.text):
		context.issues.append(
			CompilationIssue(
				issue_code='SENSITIVE_SELECT_REQUIRES_PARAMETER_MAPPING',
				explanation=(
					'The selected option contains a Browser Use secret placeholder and requires an explicit runtime parameter mapping.'
				),
				history_location=context.location,
			)
		)
		return _manual_review_outcome('A redacted secret must never be compiled as a literal select option.')

	if context.element_used is not None and context.element_used.html_tag != 'select':
		return _agentic_outcome(
			'The recorded target is not a native select element, so its custom dropdown behavior requires agentic handling.'
		)

	return _build_element_action_candidate(context, PlaywrightSelectAction(option_text=params.text))


def _validate_source_action(
	context: _ActionCompilationContext,
	*,
	model: type[_SourceActionModel],
	accepted_fields: frozenset[str],
	issue_code: str,
	explanation: str,
) -> _SourceActionModel | None:
	unexpected_fields = sorted(set(context.action_arguments) - accepted_fields)
	if unexpected_fields:
		context.issues.append(
			CompilationIssue(
				issue_code='ACTION_SCHEMA_DRIFT',
				explanation=(
					f'The {context.action_name!r} action contains unsupported fields: '
					f'{", ".join(unexpected_fields)}. Sanitized values remain in the ordered audit and the full values '
					'remain in the source history; this action will not be compiled.'
				),
				history_location=context.location,
			)
		)
		return None

	try:
		return model.model_validate(dict(context.action_arguments), strict=True)
	except ValidationError:
		context.issues.append(
			CompilationIssue(
				issue_code=issue_code,
				explanation=explanation,
				history_location=context.location,
			)
		)
		return None


def _build_element_action_candidate(
	context: _ActionCompilationContext,
	playwright_action: CachedPlaywrightAction,
) -> _CompilationOutcome:
	if context.element_used is None:
		context.issues.append(
			CompilationIssue(
				issue_code='MISSING_REQUIRED_TARGET_EVIDENCE',
				explanation=(f'The {context.action_name!r} action has no interacted-element evidence for generating a locator.'),
				history_location=context.location,
			)
		)
		return _agentic_outcome(f'No recorded element evidence is available for a deterministic {context.action_name!r} locator.')

	locator_options = build_locator_options(context.element_used)
	if not locator_options:
		context.issues.append(
			CompilationIssue(
				issue_code='NO_LOCATOR_CANDIDATE',
				explanation='Recorded element evidence did not produce a safe Playwright locator candidate.',
				history_location=context.location,
			)
		)
		return _agentic_outcome(f'The recorded element evidence cannot produce a safe locator for {context.action_name!r}.')

	highest_ranked_locator = locator_options[0].playwright_command
	candidate = DeterministicStepCandidate(
		candidate_number=context.candidate_number,
		source_step_number=context.step_number,
		playwright_action=playwright_action,
		playwright_locator_options=locator_options,
		highest_ranked_unvalidated_locator=highest_ranked_locator,
		chosen_playwright_locator=None,
		playwright_code_preview=_build_playwright_preview(highest_ranked_locator, playwright_action),
	)
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION,
			included_in_deterministic_candidates=True,
			deterministic_candidate_number=context.candidate_number,
			explanation=(
				f'The {context.action_name!r} action executed without a reported error and has evidence-derived '
				'locator candidates. A fresh Playwright replay must validate the chosen locator and action effect.'
			),
		),
		candidate,
	)


def _build_playwright_preview(locator: str, action: CachedPlaywrightAction) -> str | None:
	if isinstance(action, PlaywrightAction):
		return f'page.{locator}.{action.action_type}({python_string_literal(action.input_text)})'
	if isinstance(action, PlaywrightClickAction):
		return f'page.{locator}.click()'
	if isinstance(action, PlaywrightSelectAction):
		if action.option_match == 'unresolved':
			return None
		return f'page.{locator}.select_option({action.option_match}={python_string_literal(action.option_text)})'
	raise TypeError(f'Unsupported Playwright action model: {type(action).__name__}')


def _manual_review_outcome(explanation: str) -> _CompilationOutcome:
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.MANUAL_REVIEW_REQUIRED,
			included_in_deterministic_candidates=False,
			explanation=explanation,
		),
		None,
	)


def _agentic_outcome(explanation: str) -> _CompilationOutcome:
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING,
			included_in_deterministic_candidates=False,
			explanation=explanation,
		),
		None,
	)


_ACTION_COMPILERS: dict[str, _ActionCompiler] = {
	'input': _compile_input_action,
	'click': _compile_click_action,
	'select_dropdown': _compile_select_action,
}


def _sanitise_audit_value(value: Any, *, argument_name: str | None = None) -> Any:
	"""Preserve unsupported semantics without copying obvious credential values."""
	if argument_name is not None and _SENSITIVE_ARGUMENT_NAME_PATTERN.search(argument_name):
		return '<redacted>'
	if isinstance(value, Mapping):
		return {str(key): _sanitise_audit_value(nested_value, argument_name=str(key)) for key, nested_value in value.items()}
	if isinstance(value, list):
		return [_sanitise_audit_value(item) for item in value]
	return value
