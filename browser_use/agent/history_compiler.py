from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from browser_use.agent.history_cache.compiler import compile_history_data
from browser_use.agent.history_cache.locators import (
	MAX_LOCATOR_VALUE_LENGTH,
	MAX_XPATH_LENGTH,
	build_locator_options,
)
from browser_use.agent.history_cache.models import (
	CACHE_FORMAT_VERSION,
	BrowserAction,
	BrowserActionExecutionStatus,
	BrowserActionResult,
	BrowserUseActionCache,
	CachedAutomationDecision,
	CachedAutomationDecisionStatus,
	CachedPlaywrightAction,
	CacheModel,
	CacheStatus,
	CompilationIssue,
	ConversionSummary,
	DeterministicStepCandidate,
	ElementUsed,
	HistoryCompilationError,
	HistoryLocation,
	LocatorValidation,
	ObservedStep,
	OriginalRun,
	PageBeforeActionBatch,
	PlaywrightAction,
	PlaywrightClickAction,
	PlaywrightLocatorOption,
	PlaywrightSelectAction,
	RedundancyCheck,
	RedundancyStatus,
	SourceHistory,
)

logger = logging.getLogger(__name__)

__all__ = (
	'CACHE_FORMAT_VERSION',
	'MAX_LOCATOR_VALUE_LENGTH',
	'MAX_XPATH_LENGTH',
	'BrowserAction',
	'BrowserActionExecutionStatus',
	'BrowserActionResult',
	'BrowserUseActionCache',
	'CachedAutomationDecision',
	'CachedAutomationDecisionStatus',
	'CachedPlaywrightAction',
	'CacheModel',
	'CacheStatus',
	'CompilationIssue',
	'ConversionSummary',
	'DeterministicStepCandidate',
	'ElementUsed',
	'HistoryCompilationError',
	'HistoryLocation',
	'LocatorValidation',
	'ObservedStep',
	'OriginalRun',
	'PageBeforeActionBatch',
	'PlaywrightAction',
	'PlaywrightClickAction',
	'PlaywrightLocatorOption',
	'PlaywrightSelectAction',
	'RedundancyCheck',
	'RedundancyStatus',
	'SourceHistory',
	'build_locator_options',
	'compile_history_data',
	'compile_history_to_action_cache',
)


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
		'Compiled Browser Use action cache: observed=%d candidates=%d agentic=%d unsupported=%d manual_review=%d '
		'excluded_terminal=%d excluded_failed=%d excluded_not_executed=%d withheld_source_run=%d '
		'redundancy_candidates=%d issues=%d path=%s',
		summary.total_observed_steps,
		summary.deterministic_candidates,
		summary.requires_agentic_handling,
		summary.unsupported_steps,
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
