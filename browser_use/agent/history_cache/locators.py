from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from browser_use.agent.history_cache.models import (
	CompilationIssue,
	ElementUsed,
	HistoryLocation,
	PlaywrightLocatorOption,
)

MAX_LOCATOR_VALUE_LENGTH = 512
MAX_XPATH_LENGTH = 4096

_HTML_TAG_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9-]*$')
_DYNAMIC_PREFIX_PATTERN = re.compile(r'^(?:ember|react|vue|ng|mui|chakra|radix|headlessui)[-_]?\d', re.IGNORECASE)
_UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.IGNORECASE)
_CONTROL_CHARACTER_PATTERN = re.compile(r'[\x00-\x1f\x7f]')

_LOCATOR_ATTRIBUTE_PRIORITY: tuple[tuple[str, int], ...] = (
	('data-testid', 100),
	('data-test-id', 98),
	('data-test', 98),
	('data-cy', 98),
	('data-qa', 98),
	('id', 92),
	('name', 84),
	('title', 68),
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
		'title',
		'role',
		'type',
	}
)


def normalise_element(
	raw_element: Any,
	location: HistoryLocation,
	issues: list[CompilationIssue],
) -> ElementUsed | None:
	"""Reduce a Browser Use interacted element to locator-relevant evidence."""
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


def build_locator_options(element: ElementUsed) -> list[PlaywrightLocatorOption]:
	"""Build ranked, evidence-derived Playwright locator candidates for an element."""
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


def python_string_literal(value: str) -> str:
	"""Return a JSON-compatible quoted string suitable for Playwright previews."""
	return json.dumps(value, ensure_ascii=False)


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
	return python_string_literal(value)


def _readable_locator_name(attribute_name: str) -> str:
	return {
		'data-testid': 'test_id',
		'data-test-id': 'data_test_id_attribute',
		'data-test': 'data_test_attribute',
		'data-cy': 'data_cy_attribute',
		'data-qa': 'data_qa_attribute',
		'id': 'id_attribute',
		'name': 'name_attribute',
		'title': 'title_attribute',
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
		# Do not reject meaningful application-defined names merely because they
		# contain a numeric prefix.
		if has_alpha and len(segment) >= 8 and digit_count >= 3 and vowel_count <= 1:
			return True
		if has_alpha and vowel_count == 0 and len(segment) >= 6:
			return True
	return False
