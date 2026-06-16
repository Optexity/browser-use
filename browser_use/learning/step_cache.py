"""Learn deterministic Playwright steps from browser-use agent history.

After an agentic run, this module inspects the action history, drops exploratory
or failed steps, derives stable Playwright locators from the DOM elements the agent
actually interacted with, and emits a cache that can be converted into an Optexity
automation schema.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from browser_use.agent.views import AgentHistoryList

logger = logging.getLogger(__name__)

# Actions that are useful when converted to deterministic Optexity steps.
DETERMINISTIC_ACTIONS = frozenset({'input', 'click'})

# Exploration / bookkeeping actions that should not be cached.
SKIP_ACTIONS = frozenset(
	{
		'done',
		'scroll',
		'navigate',
		'go_back',
		'search',
		'extract',
		'send_keys',
		'switch',
		'close',
		'wait',
		'upload_file',
		'select_dropdown',
		'dropdown_options',
		'write_file',
		'read_file',
		'replace_file',
		'evaluate',
		'find_text',
	}
)


@dataclass
class CachedStep:
	action_type: str
	command: str
	prompt_instructions: str
	input_text: str | None = None
	locator_kind: str | None = None
	locator_score: int | None = None
	element_hash: int | None = None
	source_step: int = 0
	next_goal: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SkippedStep:
	source_step: int
	action_type: str
	reason: str
	params: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


class LocatorBuilder:
	"""Build stable Playwright locator strings from interacted DOM elements."""

	_DYNAMIC_PREFIX_RE = re.compile(
		r'^(?::r[0-9a-z]*:?$|react-|ember\d|radix-|headlessui-|jss\d|sc-|css-[a-z0-9]+$|emotion-)',
		re.IGNORECASE,
	)
	_UUID_RE = re.compile(
		r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
		re.IGNORECASE,
	)

	@staticmethod
	def _quote(value: str, max_len: int = 120) -> str:
		collapsed = ' '.join(value.split())
		escaped = collapsed.replace('\\', '\\\\').replace('"', '\\"')
		if len(escaped) > max_len:
			escaped = escaped[: max_len - 3] + '...'
		return f'"{escaped}"'

	@classmethod
	def _looks_dynamic(cls, value: str) -> bool:
		if not value:
			return True
		v = value.strip()
		if cls._UUID_RE.search(v) or cls._DYNAMIC_PREFIX_RE.match(v):
			return True
		if re.search(r'\d{4,}', v):
			return True
		for seg in re.split(r'[\s_\-]+', v):
			if len(seg) < 5:
				continue
			digits = sum(c.isdigit() for c in seg)
			has_alpha = any(c.isalpha() for c in seg)
			vowels = sum(c in 'aeiouAEIOU' for c in seg)
			if has_alpha and digits >= 2:
				return True
			if has_alpha and vowels == 0 and len(seg) >= 6:
				return True
		return False

	@staticmethod
	def _css_attr(tag: str, attr: str, value: str) -> str:
		return f"{tag}[{attr}='{value}']"

	@classmethod
	def _element_signals(cls, element: Any) -> tuple[str, dict[str, str], str, str]:
		if isinstance(element, dict):
			tag = (element.get('node_name') or '*').lower()
			attrs = element.get('attributes') or {}
			ax_name = (element.get('ax_name') or '').strip()
			xpath = (element.get('x_path') or element.get('xpath') or '').strip()
			return tag, attrs, ax_name, xpath

		tag = (getattr(element, 'node_name', None) or '*').lower()
		attrs = getattr(element, 'attributes', None) or {}
		ax_name = (getattr(element, 'ax_name', None) or '').strip()
		xpath = (getattr(element, 'x_path', None) or getattr(element, 'xpath', None) or '').strip()
		return tag, attrs, ax_name, xpath

	@classmethod
	def scored_candidates(cls, element: Any) -> list[tuple[int, str, str]]:
		quote = cls._quote
		tag, attrs, ax_name, xpath = cls._element_signals(element)
		text = ax_name if 0 < len(ax_name) <= 60 else ''
		role = (attrs.get('role') or '').strip()

		candidates: list[tuple[int, str, str]] = []

		for attr in ('data-testid', 'data-test-id', 'data-test', 'data-cy', 'data-qa'):
			val = (attrs.get(attr) or '').strip()
			if val and not cls._looks_dynamic(val):
				if attr == 'data-testid':
					candidates.append((100, 'test-id', f'get_by_test_id({quote(val)})'))
				else:
					candidates.append((98, 'test-id', f'locator({quote(cls._css_attr(tag, attr, val), 400)})'))

		el_id = (attrs.get('id') or '').strip()
		if el_id and not cls._looks_dynamic(el_id):
			if re.match(r'^[A-Za-z][\w-]*$', el_id):
				id_sel = f'#{el_id}'
			else:
				id_sel = cls._css_attr(tag, 'id', el_id)
			candidates.append((92, 'id', f'locator({quote(id_sel, 400)})'))

		nm = (attrs.get('name') or '').strip()
		if nm and not cls._looks_dynamic(nm):
			candidates.append((84, 'name', f'locator({quote(cls._css_attr(tag, "name", nm), 400)})'))

		aria_label = (attrs.get('aria-label') or '').strip()
		if aria_label and not cls._looks_dynamic(aria_label):
			candidates.append((76, 'aria-label', f'get_by_label({quote(aria_label)})'))

		if role and text and not cls._looks_dynamic(text):
			candidates.append((72, 'role+name', f'get_by_role({quote(role)}, name={quote(text)})'))

		placeholder = (attrs.get('placeholder') or '').strip()
		if placeholder and not cls._looks_dynamic(placeholder):
			candidates.append((64, 'placeholder', f'get_by_placeholder({quote(placeholder)})'))

		stable_classes = [c for c in (attrs.get('class') or '').split() if c and not cls._looks_dynamic(c)]
		if stable_classes:
			sel = tag + ''.join(f'.{c}' for c in stable_classes[:3])
			score = 50 + min(len(stable_classes), 3) * 3
			if text:
				candidates.append((score + 6, 'css+text', f'locator({quote(sel, 400)}, has_text={quote(text)})'))
			else:
				candidates.append((score, 'css', f'locator({quote(sel, 400)})'))

		if text:
			if role:
				candidates.append((40, 'role+text', f'get_by_role({quote(role)}, name={quote(text)})'))
			else:
				candidates.append((38, 'text', f'get_by_text({quote(text)})'))

		if xpath:
			candidates.append((10, 'xpath', f'locator({quote("xpath=" + xpath, 400)})'))

		candidates.sort(key=lambda c: c[0], reverse=True)
		return candidates

	@classmethod
	def best_locator(cls, element: Any) -> tuple[str, str, int] | None:
		candidates = cls.scored_candidates(element)
		if not candidates:
			return None
		score, kind, locator = candidates[0]
		return f'{locator}.first', kind, score


class StepCache:
	"""Extract deterministic steps from a browser-use agent run."""

	def __init__(self, start_url: str | None = None, task: str | None = None):
		self.start_url = start_url
		self.task = task
		self.deterministic_steps: list[CachedStep] = []
		self.skipped_steps: list[SkippedStep] = []

	def process_history(self, history: AgentHistoryList) -> StepCache:
		self.deterministic_steps = []
		self.skipped_steps = []
		seen_signatures: set[str] = set()
		last_click_hash: int | None = None

		for step_idx, history_item in enumerate(history.history, start=1):
			if not history_item.model_output:
				continue

			interacted_elements = history_item.state.interacted_element or []
			next_goal = history_item.model_output.next_goal
			results = history_item.result or []

			for action_idx, action in enumerate(history_item.model_output.action):
				action_dict = action.model_dump(exclude_none=True, mode='json')
				if not action_dict:
					continue

				action_name = next(iter(action_dict))
				params = action_dict[action_name] or {}
				result = results[action_idx] if action_idx < len(results) else None
				element = interacted_elements[action_idx] if action_idx < len(interacted_elements) else None

				if result and result.error:
					self.skipped_steps.append(
						SkippedStep(step_idx, action_name, f'failed: {result.error}', params)
					)
					continue

				if action_name in SKIP_ACTIONS:
					self.skipped_steps.append(
						SkippedStep(step_idx, action_name, 'exploratory or non-deterministic action', params)
					)
					continue

				if action_name not in DETERMINISTIC_ACTIONS:
					self.skipped_steps.append(
						SkippedStep(step_idx, action_name, 'unsupported action type for caching', params)
					)
					continue

				if element is None:
					self.skipped_steps.append(
						SkippedStep(step_idx, action_name, 'no interacted element recorded', params)
					)
					continue

				locator_info = LocatorBuilder.best_locator(element)
				if locator_info is None:
					self.skipped_steps.append(
						SkippedStep(step_idx, action_name, 'could not derive stable locator', params)
					)
					continue

				command, locator_kind, locator_score = locator_info
				element_hash = element.get('element_hash') if isinstance(element, dict) else getattr(element, 'element_hash', None)

				if action_name == 'input':
					text = params.get('text', '')
					signature = f'input:{element_hash}:{text}'
					if signature in seen_signatures:
						self.skipped_steps.append(
							SkippedStep(step_idx, action_name, 'duplicate input on same element', params)
						)
						continue

					# Drop focus-click immediately before typing into the same field.
					if last_click_hash is not None and last_click_hash == element_hash:
						if self.deterministic_steps and self.deterministic_steps[-1].action_type == 'click_element':
							removed = self.deterministic_steps.pop()
							self.skipped_steps.append(
								SkippedStep(
									removed.source_step,
									'click',
									'redundant focus click before input',
									{'index': params.get('index')},
								)
							)

					prompt = self._prompt_for_input(element, next_goal, text)
					self.deterministic_steps.append(
						CachedStep(
							action_type='input_text',
							command=command,
							prompt_instructions=prompt,
							input_text=text,
							locator_kind=locator_kind,
							locator_score=locator_score,
							element_hash=element_hash,
							source_step=step_idx,
							next_goal=next_goal,
						)
					)
					seen_signatures.add(signature)
					last_click_hash = None

				elif action_name == 'click':
					signature = f'click:{element_hash}'
					if signature in seen_signatures:
						self.skipped_steps.append(
							SkippedStep(step_idx, action_name, 'duplicate click on same element', params)
						)
						continue

					prompt = self._prompt_for_click(element, next_goal)
					self.deterministic_steps.append(
						CachedStep(
							action_type='click_element',
							command=command,
							prompt_instructions=prompt,
							locator_kind=locator_kind,
							locator_score=locator_score,
							element_hash=element_hash,
							source_step=step_idx,
							next_goal=next_goal,
						)
					)
					seen_signatures.add(signature)
					last_click_hash = element_hash

		return self

	@staticmethod
	def _label_from_element(element: Any) -> str:
		_, attrs, ax_name, _ = LocatorBuilder._element_signals(element)
		for key in ('aria-label', 'placeholder', 'name', 'id'):
			val = (attrs.get(key) or '').strip()
			if val:
				return val
		return ax_name or 'element'

	@classmethod
	def _prompt_for_input(cls, element: Any, next_goal: str | None, text: str) -> str:
		label = cls._label_from_element(element)
		if next_goal:
			return next_goal
		return f'Enter {text!r} in {label}'

	@classmethod
	def _prompt_for_click(cls, element: Any, next_goal: str | None) -> str:
		label = cls._label_from_element(element)
		if next_goal:
			return next_goal
		return f'Click {label}'

	def to_dict(self, history_step_count: int | None = None) -> dict[str, Any]:
		return {
			'source': 'browser-use step cache',
			'start_url': self.start_url,
			'task': self.task,
			'total_history_steps': history_step_count,
			'deterministic_step_count': len(self.deterministic_steps),
			'skipped_step_count': len(self.skipped_steps),
			'deterministic_steps': [step.to_dict() for step in self.deterministic_steps],
			'skipped_steps': [step.to_dict() for step in self.skipped_steps],
		}

	def save(self, path: str | Path, history_step_count: int | None = None) -> Path:
		path_obj = Path(path)
		path_obj.parent.mkdir(parents=True, exist_ok=True)
		with open(path_obj, 'w', encoding='utf-8') as f:
			json.dump(self.to_dict(history_step_count=history_step_count), f, indent=2)
		logger.info(
			'Saved %d deterministic step(s) and %d skipped step(s) to %s',
			len(self.deterministic_steps),
			len(self.skipped_steps),
			path_obj,
		)
		return path_obj


def save_step_cache(
	history: AgentHistoryList,
	path: str | Path,
	start_url: str | None = None,
	task: str | None = None,
) -> Path:
	cache = StepCache(start_url=start_url, task=task)
	cache.process_history(history)
	return cache.save(path, history_step_count=len(history.history))


def build_optexity_automation(
	cache: StepCache | dict[str, Any],
	url: str | None = None,
) -> dict[str, Any]:
	"""Convert cached steps into an Optexity automation JSON document."""
	if isinstance(cache, dict):
		steps = cache.get('deterministic_steps', [])
		url = url or cache.get('start_url')
	else:
		steps = [step.to_dict() for step in cache.deterministic_steps]
		url = url or cache.start_url

	nodes = []
	for step in steps:
		action_type = step['action_type']
		payload: dict[str, Any] = {
			'command': step['command'],
			'prompt_instructions': step.get('prompt_instructions', ''),
			'skip_command': False,
			'skip_prompt': True,
		}
		if action_type == 'input_text':
			payload['input_text'] = step.get('input_text', '')
			interaction = {'input_text': payload}
		elif action_type == 'click_element':
			interaction = {'click_element': payload}
		else:
			continue

		nodes.append({'type': 'action_node', 'interaction_action': interaction})

	return {
		'url': url,
		'parameters': {'input_parameters': {}, 'generated_parameters': {}},
		'nodes': nodes,
	}


def cache_to_optexity_file(
	cache_path: str | Path,
	output_path: str | Path,
	url: str | None = None,
) -> Path:
	with open(cache_path, encoding='utf-8') as f:
		cache_data = json.load(f)
	automation = build_optexity_automation(cache_data, url=url)
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	with open(output, 'w', encoding='utf-8') as f:
		json.dump(automation, f, indent=2)
	return output


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description='Build step cache or Optexity automation from agent history')
	parser.add_argument('--history', required=True, help='Path to AgentHistory.json')
	parser.add_argument('--cache-output', help='Where to write step_cache.json')
	parser.add_argument('--automation-output', help='Where to write Optexity automation JSON')
	parser.add_argument('--url', help='Starting URL for the automation')
	parser.add_argument('--task', help='Original natural-language task')
	args = parser.parse_args()

	from browser_use.agent.views import AgentOutput

	history = AgentHistoryList.load_from_file(args.history, AgentOutput)
	cache = StepCache(start_url=args.url, task=args.task)
	cache.process_history(history)

	if args.cache_output:
		cache.save(args.cache_output)
		print(f'Wrote cache to {args.cache_output}')

	if args.automation_output:
		automation = build_optexity_automation(cache, url=args.url)
		with open(args.automation_output, 'w', encoding='utf-8') as f:
			json.dump(automation, f, indent=2)
		print(f'Wrote automation to {args.automation_output}')


if __name__ == '__main__':
	main()
