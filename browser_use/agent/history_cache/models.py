from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CACHE_FORMAT_VERSION = '1.3'


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
	READY_FOR_REPLAY_VALIDATION = 'ready_for_replay_validation'
	REQUIRES_AGENTIC_HANDLING = 'requires_agentic_handling'
	UNSUPPORTED_ACTION = 'unsupported_action'
	DETERMINISTIC_ADAPTER_REQUIRED = 'deterministic_adapter_required'
	MANUAL_REVIEW_REQUIRED = 'manual_review_required'
	EXCLUDE_TERMINAL_ACTION = 'exclude_terminal_action'
	EXCLUDE_OBSERVATION_ACTION = 'exclude_observation_action'
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
	DRAFT_REQUIRES_REPLAY_VALIDATION = 'draft_requires_replay_validation'
	DRAFT_REQUIRES_AGENTIC_HANDLING = 'draft_requires_agentic_handling'
	DRAFT_REQUIRES_DETERMINISTIC_ADAPTER = 'draft_requires_deterministic_adapter'
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
	unhandled_action_arguments: dict[str, Any] = Field(default_factory=dict)


class BrowserActionResult(CacheModel):
	status: BrowserActionExecutionStatus
	has_reported_error: bool = False
	reported_success: bool | None = None
	judge_verdict: bool | None = None
	judge_failure_reason: str | None = None


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
	def validate_candidate_reference(self) -> CachedAutomationDecision:
		if self.included_in_deterministic_candidates != (self.deterministic_candidate_number is not None):
			raise ValueError('included_in_deterministic_candidates must match whether deterministic_candidate_number is present')
		candidate_decisions = {
			CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION,
			CachedAutomationDecisionStatus.READY_FOR_REPLAY_VALIDATION,
		}
		if self.included_in_deterministic_candidates != (self.decision in candidate_decisions):
			raise ValueError('Only replay candidates may reference deterministic_step_candidates')
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
	"""Backward-compatible input action stored by cache format 1.0."""

	action_type: Literal['fill', 'type']
	input_text: str


class PlaywrightClickAction(CacheModel):
	action_type: Literal['click'] = 'click'


class PlaywrightSelectAction(CacheModel):
	action_type: Literal['select_option'] = 'select_option'
	option_text: str
	option_match: Literal['unresolved', 'label', 'value'] = 'unresolved'


class PlaywrightUploadAction(CacheModel):
	action_type: Literal['upload_file'] = 'upload_file'
	file_path: str = Field(min_length=1, max_length=4096)


class PlaywrightScrollAction(CacheModel):
	action_type: Literal['scroll_element'] = 'scroll_element'
	down: bool = True
	pages: float = Field(gt=0, le=10)


CachedPlaywrightAction = Annotated[
	PlaywrightAction | PlaywrightClickAction | PlaywrightSelectAction | PlaywrightUploadAction | PlaywrightScrollAction,
	Field(discriminator='action_type'),
]


class DirectSleepAction(CacheModel):
	action_type: Literal['sleep'] = 'sleep'
	requested_seconds: int = Field(ge=0, le=30)
	replay_seconds: int = Field(ge=0, le=30)

	@model_validator(mode='after')
	def validate_browser_use_wait_parity(self) -> DirectSleepAction:
		if self.replay_seconds != min(max(self.requested_seconds - 1, 0), 30):
			raise ValueError('replay_seconds must match Browser Use wait execution semantics')
		return self


class DirectGoBackAction(CacheModel):
	action_type: Literal['go_back'] = 'go_back'


class DirectNavigateAction(CacheModel):
	action_type: Literal['navigate'] = 'navigate'
	url: str = Field(min_length=1)
	new_tab: bool = False


class DirectSearchAction(CacheModel):
	action_type: Literal['search'] = 'search'
	query: str = Field(min_length=1, max_length=4096)
	engine: Literal['duckduckgo', 'google', 'bing']
	url: str = Field(min_length=1, max_length=8192)


class DirectScrollAction(CacheModel):
	action_type: Literal['scroll'] = 'scroll'
	down: bool = True
	pages: float = Field(gt=0, le=10)


class DirectSendKeysAction(CacheModel):
	action_type: Literal['send_keys'] = 'send_keys'
	keys: str = Field(min_length=1, max_length=256)


class DirectFindTextAction(CacheModel):
	action_type: Literal['find_text'] = 'find_text'
	text: str = Field(min_length=1, max_length=4096)


CachedDirectAction = Annotated[
	DirectSleepAction
	| DirectGoBackAction
	| DirectNavigateAction
	| DirectSearchAction
	| DirectScrollAction
	| DirectSendKeysAction
	| DirectFindTextAction,
	Field(discriminator='action_type'),
]


class DeterministicStepCandidate(CacheModel):
	candidate_kind: Literal['locator'] = 'locator'
	candidate_number: int = Field(ge=1)
	source_step_number: int = Field(ge=1)
	playwright_action: CachedPlaywrightAction
	playwright_locator_options: list[PlaywrightLocatorOption]
	highest_ranked_unvalidated_locator: str
	chosen_playwright_locator: str | None = None
	playwright_code_preview: str | None

	@model_validator(mode='after')
	def validate_locator_options(self) -> DeterministicStepCandidate:
		if not self.playwright_locator_options:
			raise ValueError('A deterministic step candidate requires at least one locator option')
		if self.highest_ranked_unvalidated_locator != self.playwright_locator_options[0].playwright_command:
			raise ValueError('highest_ranked_unvalidated_locator must be the first ranked locator option')
		available_commands = {option.playwright_command for option in self.playwright_locator_options}
		if self.chosen_playwright_locator is not None and self.chosen_playwright_locator not in available_commands:
			raise ValueError('chosen_playwright_locator must reference a generated locator option')
		if (
			isinstance(self.playwright_action, PlaywrightSelectAction)
			and self.playwright_action.option_match == 'unresolved'
			and self.playwright_code_preview is not None
		):
			raise ValueError('An unresolved select action cannot expose runnable Playwright code')
		return self


class DirectActionCandidate(CacheModel):
	candidate_kind: Literal['direct'] = 'direct'
	candidate_number: int = Field(ge=1)
	source_step_number: int = Field(ge=1)
	direct_action: CachedDirectAction


CachedStepCandidate = Annotated[
	DeterministicStepCandidate | DirectActionCandidate,
	Field(discriminator='candidate_kind'),
]


class ConversionSummary(CacheModel):
	total_observed_steps: int = Field(ge=0)
	deterministic_candidates: int = Field(ge=0)
	requires_agentic_handling: int = Field(ge=0)
	manual_review_steps: int = Field(ge=0)
	excluded_terminal_steps: int = Field(ge=0)
	excluded_failed_steps: int = Field(ge=0)
	excluded_not_executed_steps: int = Field(ge=0)
	withheld_source_run_steps: int = Field(ge=0)
	unsupported_steps: int = Field(default=0, ge=0)
	deterministic_adapter_required_steps: int = Field(default=0, ge=0)
	excluded_observation_steps: int = Field(default=0, ge=0)
	suspected_redundant_steps: int = Field(ge=0)
	issues_found: int = Field(ge=0)

	@model_validator(mode='after')
	def validate_step_count(self) -> ConversionSummary:
		classified_steps = (
			self.deterministic_candidates
			+ self.requires_agentic_handling
			+ self.manual_review_steps
			+ self.excluded_terminal_steps
			+ self.excluded_failed_steps
			+ self.excluded_not_executed_steps
			+ self.withheld_source_run_steps
			+ self.unsupported_steps
			+ self.deterministic_adapter_required_steps
			+ self.excluded_observation_steps
		)
		if classified_steps != self.total_observed_steps:
			raise ValueError('Every observed step must have exactly one cached-automation decision')
		return self


class BrowserUseActionCache(CacheModel):
	cache_format_version: Literal['1.0', '1.1', '1.2', '1.3'] = CACHE_FORMAT_VERSION
	cache_status: CacheStatus
	source_history: SourceHistory
	original_run: OriginalRun
	issues: list[CompilationIssue]
	all_observed_steps: list[ObservedStep]
	deterministic_step_candidates: list[CachedStepCandidate]
	conversion_summary: ConversionSummary

	@model_validator(mode='before')
	@classmethod
	def add_legacy_locator_candidate_kind(cls, value: Any) -> Any:
		"""Keep cache formats 1.0 and 1.1 readable after adding direct candidates."""
		if not isinstance(value, dict):
			return value
		version = value.get('cache_format_version')
		candidates = value.get('deterministic_step_candidates')
		if not isinstance(candidates, list):
			return value

		normalised_candidates = []
		for candidate in candidates:
			if isinstance(candidate, dict):
				if version in {'1.0', '1.1'} and 'candidate_kind' not in candidate:
					candidate = {'candidate_kind': 'locator', **candidate}
				if version == '1.2' and candidate.get('candidate_kind') == 'direct':
					direct_action = candidate.get('direct_action')
					if isinstance(direct_action, dict) and direct_action.get('action_type') == 'sleep':
						seconds = direct_action.get('seconds')
						if isinstance(seconds, int) and not isinstance(seconds, bool):
							candidate = {
								**candidate,
								'direct_action': {
									'action_type': 'sleep',
									'requested_seconds': seconds,
									'replay_seconds': min(max(seconds - 1, 0), 30),
								},
							}
			normalised_candidates.append(candidate)
		return {**value, 'deterministic_step_candidates': normalised_candidates}

	@model_validator(mode='after')
	def validate_cross_references(self) -> BrowserUseActionCache:
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
			expected_decision = (
				CachedAutomationDecisionStatus.READY_FOR_REPLAY_VALIDATION
				if isinstance(candidate, DirectActionCandidate)
				else CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION
			)
			if decision.decision != expected_decision:
				raise ValueError('A cached-automation decision does not match its deterministic candidate kind')
		for candidate in self.deterministic_step_candidates:
			source_step = steps_by_number.get(candidate.source_step_number)
			if (
				source_step is None
				or source_step.cached_automation_decision.deterministic_candidate_number != candidate.candidate_number
			):
				raise ValueError('A deterministic candidate references an unknown source step')

		if self.conversion_summary != build_conversion_summary(self.all_observed_steps, self.issues):
			raise ValueError('conversion_summary does not match the cache contents')
		return self


def build_conversion_summary(
	observed_steps: list[ObservedStep],
	issues: list[CompilationIssue],
) -> ConversionSummary:
	"""Summarize the mutually exclusive cache decision assigned to every observed step."""
	decisions = [step.cached_automation_decision.decision for step in observed_steps]
	return ConversionSummary(
		total_observed_steps=len(observed_steps),
		deterministic_candidates=(
			decisions.count(CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION)
			+ decisions.count(CachedAutomationDecisionStatus.READY_FOR_REPLAY_VALIDATION)
		),
		requires_agentic_handling=decisions.count(CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING),
		manual_review_steps=decisions.count(CachedAutomationDecisionStatus.MANUAL_REVIEW_REQUIRED),
		excluded_terminal_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION),
		excluded_failed_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_FAILED_ACTION),
		excluded_not_executed_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_NOT_EXECUTED_ACTION),
		withheld_source_run_steps=decisions.count(CachedAutomationDecisionStatus.WITHHOLD_SOURCE_RUN_NOT_SUCCESSFUL),
		unsupported_steps=decisions.count(CachedAutomationDecisionStatus.UNSUPPORTED_ACTION),
		deterministic_adapter_required_steps=decisions.count(CachedAutomationDecisionStatus.DETERMINISTIC_ADAPTER_REQUIRED),
		excluded_observation_steps=decisions.count(CachedAutomationDecisionStatus.EXCLUDE_OBSERVATION_ACTION),
		suspected_redundant_steps=sum(
			step.redundancy_check.status == RedundancyStatus.SUSPECTED_REDUNDANT for step in observed_steps
		),
		issues_found=len(issues),
	)
