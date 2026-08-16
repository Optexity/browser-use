from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from browser_use.agent.history_cache.locators import build_locator_options, python_string_literal
from browser_use.agent.history_cache.models import (
	BrowserAction,
	BrowserActionExecutionStatus,
	BrowserActionResult,
	CachedAutomationDecision,
	CachedAutomationDecisionStatus,
	CachedDirectAction,
	CachedPlaywrightAction,
	CachedStepCandidate,
	CompilationIssue,
	DeterministicStepCandidate,
	DirectActionCandidate,
	DirectFindTextAction,
	DirectGoBackAction,
	DirectNavigateAction,
	DirectScrollAction,
	DirectSearchAction,
	DirectSendKeysAction,
	DirectSleepAction,
	ElementUsed,
	HistoryLocation,
	PlaywrightAction,
	PlaywrightClickAction,
	PlaywrightScrollAction,
	PlaywrightSelectAction,
	PlaywrightUploadAction,
)
from browser_use.tools.views import (
	ClickElementAction,
	GetDropdownOptionsAction,
	InputTextAction,
	NavigateAction,
	NoParamsAction,
	ScrollAction,
	SearchAction,
	SelectDropdownOptionAction,
	SendKeysAction,
	UploadFileAction,
)

_SECRET_PLACEHOLDER_PATTERN = re.compile(r'<secret>[^<>]+</secret>')
_SENSITIVE_ARGUMENT_NAME_PATTERN = re.compile(
	r'(?:password|passwd|secret|token|api[_-]?key|authorization|cookie|credential)',
	re.IGNORECASE,
)

_INPUT_ACTION_FIELDS = frozenset({'index', 'text', 'clear'})
_CLICK_ACTION_FIELDS = frozenset({'index', 'coordinate_x', 'coordinate_y'})
_SELECT_ACTION_FIELDS = frozenset({'index', 'text'})
_NAVIGATE_ACTION_FIELDS = frozenset({'url', 'new_tab'})
_GO_BACK_ACTION_FIELDS = frozenset({'description'})
_WAIT_ACTION_FIELDS = frozenset({'seconds'})
_SEARCH_ACTION_FIELDS = frozenset({'query', 'engine'})
_SCROLL_ACTION_FIELDS = frozenset({'down', 'pages', 'index'})
_SEND_KEYS_ACTION_FIELDS = frozenset({'keys'})
_FIND_TEXT_ACTION_FIELDS = frozenset({'text'})
_UPLOAD_ACTION_FIELDS = frozenset({'index', 'path'})
_EVALUATE_ACTION_FIELDS = frozenset({'code'})
_SCREENSHOT_ACTION_FIELDS = frozenset({'description'})
_DROPDOWN_OPTIONS_FIELDS = frozenset({'index'})
_COMPILED_ACTION_FIELDS = {
	'input': _INPUT_ACTION_FIELDS,
	'click': _CLICK_ACTION_FIELDS,
	'select_dropdown': _SELECT_ACTION_FIELDS,
	'navigate': _NAVIGATE_ACTION_FIELDS,
	'go_back': _GO_BACK_ACTION_FIELDS,
	'wait': _WAIT_ACTION_FIELDS,
	'search': _SEARCH_ACTION_FIELDS,
	'scroll': _SCROLL_ACTION_FIELDS,
	'send_keys': _SEND_KEYS_ACTION_FIELDS,
	'find_text': _FIND_TEXT_ACTION_FIELDS,
	'upload_file': _UPLOAD_ACTION_FIELDS,
	'evaluate': _EVALUATE_ACTION_FIELDS,
	'screenshot': _SCREENSHOT_ACTION_FIELDS,
	'dropdown_options': _DROPDOWN_OPTIONS_FIELDS,
}

_DETERMINISTIC_RUNTIME_GAP_ACTIONS = frozenset({'switch', 'close', 'write_file', 'replace_file', 'read_file'})


class _WaitAction(BaseModel):
	"""Source contract generated from Browser Use's ``wait(seconds: int)`` tool."""

	model_config = ConfigDict(extra='forbid')

	seconds: int = Field(default=3, ge=0, le=30)


class _FindTextAction(BaseModel):
	model_config = ConfigDict(extra='forbid')

	text: str = Field(min_length=1, max_length=4096)


class _EvaluateAction(BaseModel):
	model_config = ConfigDict(extra='forbid')

	code: str = Field(min_length=1, max_length=20000)


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


_CompilationOutcome = tuple[CachedAutomationDecision, CachedStepCandidate | None]
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
	elif action_name == 'search':
		action_details = {
			'query': action_arguments.get('query'),
			'engine': action_arguments.get('engine', 'duckduckgo'),
		}
	elif action_name == 'scroll':
		action_details = {
			'down': action_arguments.get('down', True),
			'pages': action_arguments.get('pages', 1.0),
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
) -> tuple[CachedAutomationDecision, CachedStepCandidate | None]:
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
		if action_name in _DETERMINISTIC_RUNTIME_GAP_ACTIONS:
			return (
				CachedAutomationDecision(
					decision=CachedAutomationDecisionStatus.DETERMINISTIC_ADAPTER_REQUIRED,
					included_in_deterministic_candidates=False,
					explanation=(
						f'Browser Use action {action_name!r} is deterministic, but the current Optexity '
						'runtime does not yet expose an equivalent typed action. It is retained for a '
						'deterministic adapter and is never eligible for LLM conversion.'
					),
				),
				None,
			)
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
	try:
		return action_compiler(context)
	except ValidationError:
		context.issues.append(
			CompilationIssue(
				issue_code='CANDIDATE_VALIDATION_FAILED',
				explanation=(
					'The recorded action is valid Browser Use history, but its replay candidate '
					'exceeds the cache contract and requires manual review.'
				),
				history_location=context.location,
			)
		)
		return _manual_review_outcome(
			'The deterministic replay candidate could not be represented safely in the current cache schema.'
		)


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
		return _deterministic_adapter_required_outcome(
			'A coordinate-only click is mechanically deterministic, but safe replay requires viewport/scale provenance '
			'and a dedicated raw-coordinate runtime action.'
		)

	if context.element_used is not None and context.element_used.html_tag == 'select':
		return _excluded_observation_outcome(
			'Browser Use treats clicking a native select as dropdown inspection. The observation is retained but is not '
			'replayed as a mutating click.'
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


def _compile_navigate_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=NavigateAction,
		accepted_fields=_NAVIGATE_ACTION_FIELDS,
		issue_code='INVALID_NAVIGATE_ACTION_ARGUMENTS',
		explanation='A navigate action requires a non-empty URL and a boolean new_tab flag.',
	)
	if params is None or not params.url:
		return _manual_review_outcome('The navigate arguments do not match the Browser Use navigation contract.')
	return _build_direct_action_candidate(
		context,
		DirectNavigateAction(url=params.url, new_tab=params.new_tab),
	)


def _compile_go_back_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=NoParamsAction,
		accepted_fields=_GO_BACK_ACTION_FIELDS,
		issue_code='INVALID_GO_BACK_ACTION_ARGUMENTS',
		explanation='A go-back action accepts no behavior-changing arguments.',
	)
	if params is None:
		return _manual_review_outcome('The go-back arguments do not match the Browser Use action contract.')
	return _build_direct_action_candidate(context, DirectGoBackAction())


def _compile_wait_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=_WaitAction,
		accepted_fields=_WAIT_ACTION_FIELDS,
		issue_code='INVALID_WAIT_ACTION_ARGUMENTS',
		explanation='A wait action requires an integer duration between 0 and 30 seconds.',
	)
	if params is None:
		return _manual_review_outcome('The wait arguments do not match the Browser Use wait-action contract.')
	return _build_direct_action_candidate(
		context,
		DirectSleepAction(
			requested_seconds=params.seconds,
			replay_seconds=min(max(params.seconds - 1, 0), 30),
		),
	)


def _compile_search_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=SearchAction,
		accepted_fields=_SEARCH_ACTION_FIELDS,
		issue_code='INVALID_SEARCH_ACTION_ARGUMENTS',
		explanation='A search action requires a non-empty query and one supported search engine.',
	)
	if params is None or not params.query:
		return _manual_review_outcome('The search arguments do not match the Browser Use search-action contract.')
	engine = params.engine.lower()
	encoded_query = urllib.parse.quote_plus(params.query)
	search_urls = {
		'duckduckgo': f'https://duckduckgo.com/?q={encoded_query}',
		'google': f'https://www.google.com/search?q={encoded_query}&udm=14',
		'bing': f'https://www.bing.com/search?q={encoded_query}',
	}
	url = search_urls.get(engine)
	if url is None:
		return _manual_review_outcome('The recorded search engine is not supported by Browser Use replay.')
	engine_name = cast(Literal['duckduckgo', 'google', 'bing'], engine)
	return _build_direct_action_candidate(
		context,
		DirectSearchAction(query=params.query, engine=engine_name, url=url),
	)


def _compile_scroll_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=ScrollAction,
		accepted_fields=_SCROLL_ACTION_FIELDS,
		issue_code='INVALID_SCROLL_ACTION_ARGUMENTS',
		explanation='A scroll action requires a positive number of viewport pages, optionally scoped to an element.',
	)
	if params is None or params.pages <= 0 or params.pages > 10:
		return _manual_review_outcome('The scroll arguments do not match the bounded Browser Use scroll contract.')
	if params.index in {None, 0}:
		return _build_direct_action_candidate(
			context,
			DirectScrollAction(down=params.down, pages=params.pages),
		)
	return _build_element_action_candidate(
		context,
		PlaywrightScrollAction(down=params.down, pages=params.pages),
	)


def _compile_send_keys_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=SendKeysAction,
		accepted_fields=_SEND_KEYS_ACTION_FIELDS,
		issue_code='INVALID_SEND_KEYS_ACTION_ARGUMENTS',
		explanation='A send-keys action requires one non-empty key or key chord string.',
	)
	if params is None or not params.keys.strip():
		return _manual_review_outcome('The send-keys arguments do not match the Browser Use action contract.')
	return _build_direct_action_candidate(context, DirectSendKeysAction(keys=params.keys))


def _compile_find_text_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=_FindTextAction,
		accepted_fields=_FIND_TEXT_ACTION_FIELDS,
		issue_code='INVALID_FIND_TEXT_ACTION_ARGUMENTS',
		explanation='A find-text action requires a non-empty text string.',
	)
	if params is None:
		return _manual_review_outcome('The find-text arguments do not match the Browser Use action contract.')
	return _build_direct_action_candidate(context, DirectFindTextAction(text=params.text))


def _compile_upload_file_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=UploadFileAction,
		accepted_fields=_UPLOAD_ACTION_FIELDS,
		issue_code='INVALID_UPLOAD_FILE_ACTION_ARGUMENTS',
		explanation='An upload-file action requires an element index and a non-empty file path.',
	)
	if params is None or params.index < 0 or not params.path:
		return _manual_review_outcome('The upload-file arguments do not match the Browser Use action contract.')
	if context.element_used is None:
		return _deterministic_adapter_required_outcome(
			'The upload action has no target evidence. A file input must be identified deterministically before replay.'
		)
	element_type = context.element_used.locator_relevant_attributes.get('type', '').lower()
	if context.element_used.html_tag != 'input' or element_type != 'file':
		return _deterministic_adapter_required_outcome(
			'Upload replay requires recorded input[type="file"] evidence; a nearby or inferred file input is not promoted.'
		)
	return _build_element_action_candidate(context, PlaywrightUploadAction(file_path=params.path))


def _compile_evaluate_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=_EvaluateAction,
		accepted_fields=_EVALUATE_ACTION_FIELDS,
		issue_code='INVALID_EVALUATE_ACTION_ARGUMENTS',
		explanation='An evaluate action requires non-empty JavaScript within the configured replay size limit.',
	)
	if params is None:
		return _manual_review_outcome('The evaluate arguments do not match the Browser Use action contract.')
	return _deterministic_adapter_required_outcome(
		'The exact JavaScript is retained as deterministic source evidence, but persistent arbitrary-code replay '
		'requires an explicit trusted-history policy and a browser-only Optexity runtime action. It is never sent to '
		'the conversion LLM.'
	)


def _compile_screenshot_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=NoParamsAction,
		accepted_fields=_SCREENSHOT_ACTION_FIELDS,
		issue_code='INVALID_SCREENSHOT_ACTION_ARGUMENTS',
		explanation='A screenshot observation does not accept behavior-changing arguments.',
	)
	if params is None:
		return _manual_review_outcome('The screenshot arguments do not match the Browser Use action contract.')
	return _excluded_observation_outcome(
		'The screenshot action only requested the next Browser Use observation and did not change the webpage.'
	)


def _compile_dropdown_options_action(context: _ActionCompilationContext) -> _CompilationOutcome:
	params = _validate_source_action(
		context,
		model=GetDropdownOptionsAction,
		accepted_fields=_DROPDOWN_OPTIONS_FIELDS,
		issue_code='INVALID_DROPDOWN_OPTIONS_ARGUMENTS',
		explanation='A dropdown-options observation requires an element index.',
	)
	if params is None or params.index < 0:
		return _manual_review_outcome('The dropdown-options arguments do not match the Browser Use contract.')
	return _excluded_observation_outcome(
		'The dropdown-options action only inspected choices and did not change the webpage. The selected option, when '
		'present, is represented by its own ordered select action.'
	)


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


def _build_direct_action_candidate(
	context: _ActionCompilationContext,
	direct_action: CachedDirectAction,
) -> _CompilationOutcome:
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.READY_FOR_REPLAY_VALIDATION,
			included_in_deterministic_candidates=True,
			deterministic_candidate_number=context.candidate_number,
			explanation=(
				f'The {context.action_name!r} action executed without a reported error and has a typed direct replay '
				'candidate. A fresh replay must still validate its effect.'
			),
		),
		DirectActionCandidate(
			candidate_number=context.candidate_number,
			source_step_number=context.step_number,
			direct_action=direct_action,
		),
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
	if isinstance(action, PlaywrightUploadAction):
		return f'page.{locator}.set_input_files({python_string_literal(action.file_path)})'
	if isinstance(action, PlaywrightScrollAction):
		direction = 1 if action.down else -1
		return (
			f'page.{locator}.evaluate("(element, args) => '
			f'element.scrollBy(0, args.direction * element.clientHeight * args.pages)", '
			f'{{"direction": {direction}, "pages": {action.pages!r}}})'
		)
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


def _deterministic_adapter_required_outcome(explanation: str) -> _CompilationOutcome:
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.DETERMINISTIC_ADAPTER_REQUIRED,
			included_in_deterministic_candidates=False,
			explanation=explanation,
		),
		None,
	)


def _excluded_observation_outcome(explanation: str) -> _CompilationOutcome:
	return (
		CachedAutomationDecision(
			decision=CachedAutomationDecisionStatus.EXCLUDE_OBSERVATION_ACTION,
			included_in_deterministic_candidates=False,
			explanation=explanation,
		),
		None,
	)


_ACTION_COMPILERS: dict[str, _ActionCompiler] = {
	'input': _compile_input_action,
	'click': _compile_click_action,
	'select_dropdown': _compile_select_action,
	'navigate': _compile_navigate_action,
	'go_back': _compile_go_back_action,
	'wait': _compile_wait_action,
	'search': _compile_search_action,
	'scroll': _compile_scroll_action,
	'send_keys': _compile_send_keys_action,
	'find_text': _compile_find_text_action,
	'upload_file': _compile_upload_file_action,
	'evaluate': _compile_evaluate_action,
	'screenshot': _compile_screenshot_action,
	'dropdown_options': _compile_dropdown_options_action,
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
