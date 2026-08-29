"""Agent step cache — a memory layer for deterministic replay.

Every agent step costs an LLM call, even though most steps (type into this
field, click that button) are deterministic once they have been solved once.
This module records what the agent actually did, classifies each step as
``deterministic``, ``redundant`` or ``terminal``, and rebuilds the
deterministic steps as Playwright locator commands and an Optexity automation
dict that can be replayed with zero LLM calls.

Typical flow:

	agent = Agent(task=..., llm=..., settings=AgentSettings(save_step_cache_path='agent_step_cache.json'))
	history = await agent.run()  # cache is written automatically at the end of the run

	cache = AgentStepCache.load_from_file('agent_step_cache.json')
	print(cache.summary())
	automation_dict = cache.to_optexity_automation_dict(url='https://example.com')
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import BaseModel, Field

from browser_use.agent.views import AgentHistoryList


class CachedElement(BaseModel):
	"""Fingerprint of the DOM element the agent interacted with.

	Contains everything needed to rebuild a deterministic Playwright locator
	for the element without seeing the page again.
	"""

	tag_name: str
	attributes: dict[str, str] = Field(default_factory=dict)
	x_path: str | None = None
	ax_name: str | None = None
	node_value: str | None = None

	def describe(self) -> str:
		"""Human readable description, used for prompt_instructions."""
		for key in ('aria-label', 'name', 'placeholder', 'id', 'title'):
			value = self.attributes.get(key)
			if value:
				return f'{self.tag_name} "{value}"'
		if self.ax_name:
			return f'{self.tag_name} "{self.ax_name}"'
		return self.tag_name


class CachedStep(BaseModel):
	"""A single agent step recorded from the agent history."""

	step_index: int
	url: str
	page_title: str | None = None
	action_name: str
	action_params: dict[str, Any] = Field(default_factory=dict)
	element: CachedElement | None = None
	# Text typed / selected for input-like actions
	text: str | None = None
	# True when a following send_keys(Enter) was folded into this input step
	press_enter: bool = False
	# Files downloaded as a result of this action (e.g. a click triggering a download)
	downloaded_files: list[str] = Field(default_factory=list)
	success: bool = True
	error: str | None = None
	is_done: bool = False
	duration_seconds: float | None = None
	classification: Literal['deterministic', 'redundant', 'terminal'] = 'redundant'
	reason: str = ''


class AgentStepCache(BaseModel):
	"""The full recorded memory of one agent run."""

	task: str
	created_at: str
	start_url: str | None = None
	total_steps: int = 0
	# Run metrics — makes the agentic vs deterministic comparison data-driven
	total_duration_seconds: float | None = None
	total_prompt_tokens: int | None = None
	total_completion_tokens: int | None = None
	total_cost: float | None = None
	steps: list[CachedStep] = Field(default_factory=list)

	# ---- accessors ----

	def deterministic_steps(self) -> list[CachedStep]:
		"""Steps that can be replayed deterministically (no LLM needed)."""
		return [step for step in self.steps if step.classification == 'deterministic']

	def redundant_steps(self) -> list[CachedStep]:
		"""Exploratory/auxiliary/failed steps that a replay should skip."""
		return [step for step in self.steps if step.classification == 'redundant']

	def terminal_steps(self) -> list[CachedStep]:
		"""Steps that mark the end of the task (e.g. done)."""
		return [step for step in self.steps if step.classification == 'terminal']

	def replay_steps(self) -> list[CachedStep]:
		"""Deterministic steps ready for replay, with consecutive duplicates removed.

		browser-use sometimes repeats an identical successful action while
		exploring (typing into the same field twice, clicking the same button
		again). Replaying those verbatim would be redundant, so keep only the
		first of each consecutive run of identical steps.
		"""
		replay: list[CachedStep] = []
		for step in self.deterministic_steps():
			if replay and self._is_duplicate_step(replay[-1], step):
				continue
			replay.append(step)
		return replay

	@staticmethod
	def _is_duplicate_step(a: CachedStep, b: CachedStep) -> bool:
		"""Two steps are duplicates when they act identically on the same element."""
		if a.action_name != b.action_name or a.text != b.text:
			return False
		if a.element is None or b.element is None:
			return False
		return (
			a.element.tag_name == b.element.tag_name
			and a.element.attributes == b.element.attributes
			and a.element.x_path == b.element.x_path
			and a.element.ax_name == b.element.ax_name
		)

	def summary(self) -> str:
		return (
			f'AgentStepCache: {self.total_steps} steps total | '
			f'{len(self.deterministic_steps())} deterministic | '
			f'{len(self.redundant_steps())} redundant | '
			f'{len(self.terminal_steps())} terminal'
		)

	# ---- persistence ----

	def save_to_file(self, filepath: str | os.PathLike) -> None:
		"""Write the cache to a JSON file."""
		with open(filepath, 'w', encoding='utf-8') as f:
			json.dump(self.model_dump(), f, indent=2)

	@classmethod
	def load_from_file(cls, filepath: str | os.PathLike) -> 'AgentStepCache':
		"""Load a cache from a JSON file."""
		with open(filepath, 'r', encoding='utf-8') as f:
			return cls.model_validate(json.load(f))

	# ---- replay builders ----

	def to_playwright_commands(self) -> list[dict[str, Any]]:
		"""Deterministic steps as an ordered list of Playwright commands.

		Each item: {index, step_index, action, command, value, url, description}.
		"""
		commands: list[dict[str, Any]] = []
		for position, step in enumerate(self.replay_steps()):
			command = playwright_command(step.element) if step.element else ''
			commands.append(
				{
					'index': position,
					'step_index': step.step_index,
					'action': step.action_name,
					'command': command,
					'value': step.text,
					'url': step.url,
					'description': step.element.describe() if step.element else None,
				}
			)
		return commands

	def to_optexity_automation_dict(self, url: str | None = None) -> dict[str, Any]:
		"""Deterministic steps as an Optexity automation dict.

		The ``command`` values are sourced from the recorded element
		fingerprints, so the resulting automation replays the run without any
		LLM reasoning. Raises ``ValueError`` if a deterministic step cannot be
		converted (e.g. no usable locator could be built).
		"""
		nodes: list[dict[str, Any]] = []
		for position, step in enumerate(self.replay_steps()):
			node = self._step_to_optexity_node(step, position)
			if node is not None:
				nodes.append(node)
		return {
			'url': url or self.start_url or (self.steps[0].url if self.steps else ''),
			'parameters': {
				'input_parameters': {},
				'generated_parameters': {},
			},
			'nodes': nodes,
		}

	def _step_to_optexity_node(self, step: CachedStep, position: int) -> dict[str, Any] | None:
		"""Convert one deterministic step into an Optexity action node."""
		if step.action_name == 'navigate':
			target = str(step.action_params.get('url', ''))
			if not target:
				raise ValueError(f'Step {step.step_index}: navigate without target url')
			return {
				'type': 'action_node',
				'interaction_action': {
					'go_to_url': {
						'url': target,
						'new_tab': bool(step.action_params.get('new_tab', False)),
					}
				},
			}

		command = playwright_command(step.element) if step.element else ''
		if not command:
			raise ValueError(
				f'Step {step.step_index} ({step.action_name}): could not build a playwright command from element fingerprint'
			)
		description = step.element.describe() if step.element else 'element'

		if step.action_name == 'click':
			script_node = _inline_plaintext_download_node(step, _landing_url_after(self.steps, step))
			if script_node is not None:
				# The recorded click targeted a plaintext URL the browser renders
				# inline (it never triggered a real download). Emit a
				# deterministic fetch + save_download script instead of a
				# click_element so the replay produces an actual task download.
				return script_node
			click_action: dict[str, Any] = {
				'command': command,
				'prompt_instructions': f'Click {description}',
			}
			if step.downloaded_files:
				# This click triggered a download during the recorded run
				click_action['expect_download'] = True
				click_action['download_filename'] = step.downloaded_files[0]
			interaction: dict[str, Any] = {'click_element': click_action}
		elif step.action_name == 'input':
			interaction = {
				'input_text': {
					'command': command,
					'prompt_instructions': f"Enter '{step.text}' in {description}",
					'input_text': step.text or '',
					'press_enter': step.press_enter,
				}
			}
		elif step.action_name == 'select_dropdown':
			interaction = {
				'select_option': {
					'command': command,
					'prompt_instructions': f"Select '{step.text}' in {description}",
					'select_values': [step.text or ''],
				}
			}
		elif step.action_name == 'upload_file':
			interaction = {
				'upload_file': {
					'command': command,
					'file_path': step.action_params.get('path'),
					'prompt_instructions': f'Upload file in {description}',
				}
			}
		else:
			raise ValueError(f'Step {step.step_index}: unsupported deterministic action {step.action_name!r}')

		return {'type': 'action_node', 'interaction_action': interaction}


# ---- classification ----

# Actions that interact with a concrete element and can be replayed as-is.
DETERMINISTIC_ELEMENT_ACTIONS = frozenset({'click', 'input', 'select_dropdown', 'upload_file'})
# Actions that only support the agent's own reasoning (exploration, reading,
# keyboard shortcuts) and add no deterministic state change.
REDUNDANT_ACTIONS = frozenset(
	{
		'scroll',
		'send_keys',
		'get_dropdown_options',
		'extract',
		'search',
		'find_text',
		'highlight',
		'wait',
		'switch',
		'close_tab',
		'close_tabs',
	}
)


def classify_step(
	action_name: str,
	params: dict[str, Any],
	element: CachedElement | None,
	success: bool,
	is_done: bool,
	start_url: str | None = None,
) -> tuple[Literal['deterministic', 'redundant', 'terminal'], str]:
	"""Decide whether a recorded step should be replayed deterministically.

	Returns (classification, reason).
	"""
	if is_done:
		return 'terminal', 'task completion marker'

	if not success:
		return 'redundant', 'failed step — exploration attempt, not needed for replay'

	if action_name in REDUNDANT_ACTIONS:
		return 'redundant', 'exploratory/auxiliary step with no deterministic state change'

	if action_name == 'navigate':
		target = params.get('url')
		if start_url and target == start_url:
			return 'redundant', 'navigation to the start URL — already covered by the automation url'
		return 'deterministic', 'explicit navigation to a new URL'

	if action_name in DETERMINISTIC_ELEMENT_ACTIONS:
		if element is None:
			return 'redundant', 'no element fingerprint recorded (coordinate-based or index-less action)'
		return 'deterministic', f'{action_name} on a concrete element'

	return 'redundant', f'unsupported action {action_name!r}'


# ---- locator building ----

_TAG_TO_ROLE = {
	'a': 'link',
	'button': 'button',
	'select': 'combobox',
	'textarea': 'textbox',
	'option': 'option',
	'label': 'label',
	'summary': 'summary',
}


def _role_for_tag(tag_name: str, attributes: dict[str, str]) -> str | None:
	tag = tag_name.lower()
	if tag == 'input':
		input_type = attributes.get('type', 'text').lower()
		if input_type in ('checkbox',):
			return 'checkbox'
		if input_type in ('radio',):
			return 'radio'
		if input_type in ('submit', 'button', 'image', 'reset'):
			return 'button'
		return 'textbox'
	return _TAG_TO_ROLE.get(tag)


def _quote(value: str) -> str:
	"""Escape a value for a double-quoted string inside a locator expression."""
	return value.replace('\\', '\\\\').replace('"', '\\"')


def playwright_command(element: CachedElement) -> str:
	"""Build a deterministic ``page.<command>`` Playwright expression.

	Strategy, most robust first:
	  1. ``#id``                -> locator("#id").first
	  2. data-testid           -> get_by_test_id("...")
	  3. href on links         -> locator("a[href='...']").first
	  4. name on form controls -> locator("[name='...']").first
	  5. aria-label            -> get_by_label("...")
	  6. accessible name       -> get_by_role("<role>", name="...")
	  7. placeholder           -> get_by_placeholder("...")
	  8. visible text          -> get_by_text("...")
	  9. recorded xpath        -> locator("xpath=...").first
	"""
	attributes = element.attributes or {}
	tag = element.tag_name.lower()

	# 1. id — most stable when present
	element_id = attributes.get('id')
	if element_id:
		return f'locator("#{_quote(element_id)}").first'

	# 2. test id
	test_id = attributes.get('data-testid') or attributes.get('data-test-id') or attributes.get('data-cy')
	if test_id:
		return f'get_by_test_id("{_quote(test_id)}")'

	# 3. href on links — semantic and stable across text changes (e.g. dynamic
	#    counters embedded in the link's accessible name)
	href = attributes.get('href')
	if href and tag == 'a':
		return f'locator("a[href=\'{_quote(href)}\']").first'

	# 4. name attribute on form controls
	name_attr = attributes.get('name')
	if name_attr and tag in ('input', 'textarea', 'select'):
		return f'locator("[name=\'{_quote(name_attr)}\']").first'

	# 4. aria-label
	aria_label = attributes.get('aria-label')
	if aria_label:
		return f'get_by_label("{_quote(aria_label)}")'

	# 5. accessible name (from the accessibility tree)
	#    <summary> elements first: Playwright's role engine does not reliably
	#    expose them as role "summary", which resolves to a not-visible match;
	#    match them by visible text instead.
	if tag == 'summary' and element.ax_name:
		return f'get_by_text("{_quote(element.ax_name)}")'

	if element.ax_name:
		role = _role_for_tag(tag, attributes)
		if role:
			return f'get_by_role("{role}", name="{_quote(element.ax_name)}")'

	# 6. placeholder
	placeholder = attributes.get('placeholder')
	if placeholder:
		return f'get_by_placeholder("{_quote(placeholder)}")'

	# 7. visible text
	if element.ax_name:
		return f'get_by_text("{_quote(element.ax_name)}")'

	# 8. recorded xpath as a last resort
	if element.x_path:
		return f'locator("xpath={_quote(element.x_path)}").first'

	return ''


# Suffixes browsers render inline when navigated to (no Content-Disposition
# involved): a cached click on one of these re-opens the text in the tab
# instead of downloading. The Optexity automation-dict compiler replaces such
# clicks with a deterministic fetch + ctx.save_download script node so the
# replay produces a real task download. The fetched URL and filename both come
# from the cached element fingerprint, so the script is still cache-derived.
_INLINE_PLAINTEXT_SUFFIXES = frozenset({'.txt', '.txt.utf-8', '.md', '.markdown', '.log', '.rst'})


def _inline_plaintext_download_node(step: CachedStep, landing_url: str | None = None) -> dict[str, Any] | None:
	"""Compile a cached click on an inline-plaintext link into a download script node.

	Returns None when ``step`` is not such a click so callers fall back to a
	regular click_element node.

	``landing_url`` is the URL the page settled on right after this step (from
	the cache's next record). When available it is preferred as the fetch
	target: the recorded href may redirect through ``http`` (e.g. Gutenberg
	302s ``/ebooks/84.txt.utf-8`` -> ``cache/epub/84/pg84.txt``), and the
	in-page ``fetch()`` API refuses to follow the https->http downgrade,
	failing with ``TypeError: Failed to fetch``. The landing URL is
	cache-derived and fetch-safe.
	"""
	if step.action_name != 'click' or step.downloaded_files or step.element is None:
		return None
	href = (step.element.attributes or {}).get('href')
	if not href:
		return None

	parsed = urlsplit(urljoin(step.url, href))
	if not parsed.netloc or not parsed.path:
		return None
	if not parsed.path.lower().endswith(tuple(_INLINE_PLAINTEXT_SUFFIXES)):
		return None

	url = urlunsplit(parsed)
	if landing_url:
		landing_parsed = urlsplit(landing_url)
		if landing_parsed.netloc and landing_parsed.path.lower().endswith(tuple(_INLINE_PLAINTEXT_SUFFIXES)):
			url = landing_url
	filename = urlsplit(url).path.rsplit('/', 1)[-1] or parsed.path.rsplit('/', 1)[-1] or 'download.txt'
	# fetch through the live page: same-origin for the recorded run's site
	execution_code = (
		'async def code_fn(page, ctx):\n'
		'    data = await page.evaluate(\n'
		f'        """async () => {{\n'
		f'            const res = await fetch({json.dumps(url)});\n'
		f"            if (!res.ok) throw new Error('HTTP ' + res.status);\n"
		f'            return Array.from(new Uint8Array(await res.arrayBuffer()));\n'
		'        }"""\n'
		'    )\n'
		f'    await ctx.save_download({json.dumps(filename)}, content=bytes(data))\n'
	)
	return {'type': 'action_node', 'python_script_action': {'execution_code': execution_code}}


def _landing_url_after(cache: list[CachedStep], step: CachedStep) -> str | None:
	"""The URL the page settled on right after ``step`` (cache-derived).

	Searches the recorded steps in order for the first one after ``step`` whose
	URL differs. Used for download scripts: the click's href may redirect, and
	the final address is what the browser actually navigated to.
	"""
	for other in cache:
		if other.step_index > step.step_index and other.url and other.url != step.url:
			return other.url
	return None


# ---- cache building ----


def build_step_cache(
	history: AgentHistoryList,
	task: str = '',
	start_url: str | None = None,
) -> AgentStepCache:
	"""Build an :class:`AgentStepCache` from a completed agent run history.

	Each history item may contain multiple actions; interacted elements are
	index-aligned with the action list. Steps that failed or only supported the
	agent's own reasoning are kept in the cache (marked redundant) so the
	classification stays auditable.
	"""
	steps: list[CachedStep] = []
	first_url: str | None = start_url

	for item in history.history:
		output = item.model_output
		if output is None or not output.action:
			# Pseudo history items (e.g. max-steps failure marker) carry no action
			continue

		if first_url is None and item.state.url:
			first_url = item.state.url

		interacted = item.state.interacted_element or []
		for action_position, action in enumerate(output.action):
			action_data = action.model_dump(exclude_unset=True)
			if not action_data:
				continue
			action_name = next(iter(action_data.keys()))
			params = action_data.get(action_name) or {}

			result = None
			if item.result:
				result = item.result[action_position] if action_position < len(item.result) else item.result[0]
			success = bool(result is not None and result.error is None)
			error = result.error if result is not None else None
			is_done = bool(result is not None and result.is_done)
			downloaded_files = list(result.attachments) if result is not None and result.attachments else []

			recorded = interacted[action_position] if action_position < len(interacted) else None
			element = None
			if recorded is not None:
				element = CachedElement(
					tag_name=(recorded.node_name or '').lower(),
					attributes=dict(recorded.attributes or {}),
					x_path=recorded.x_path,
					ax_name=recorded.ax_name,
					node_value=recorded.node_value,
				)

			text = None
			if action_name in ('input', 'select_dropdown'):
				text = params.get('text')

			# Fold a trailing send_keys(Enter) into the previous input step so the
			# deterministic replay still submits the form (optexity press_enter).
			folded_enter = (
				action_name == 'send_keys'
				and 'Enter' in str(params.get('keys', ''))
				and bool(steps)
				and steps[-1].action_name == 'input'
				and steps[-1].success
			)
			if folded_enter:
				steps[-1].press_enter = True

			classification, reason = classify_step(
				action_name=action_name,
				params=params,
				element=element,
				success=success,
				is_done=is_done,
				start_url=first_url,
			)
			if folded_enter:
				reason = 'Enter folded into the previous input step (press_enter)'

			steps.append(
				CachedStep(
					step_index=len(steps),
					url=item.state.url,
					page_title=item.state.title,
					action_name=action_name,
					action_params=dict(params),
					element=element,
					text=text,
					downloaded_files=downloaded_files,
					success=success,
					error=error,
					is_done=is_done,
					duration_seconds=item.metadata.duration_seconds if item.metadata else None,
					classification=classification,
					reason=reason,
				)
			)

	usage = history.usage
	return AgentStepCache(
		task=task,
		created_at=datetime.now(timezone.utc).isoformat(),
		start_url=first_url,
		total_steps=len(steps),
		total_duration_seconds=history.total_duration_seconds(),
		total_prompt_tokens=getattr(usage, 'total_prompt_tokens', None),
		total_completion_tokens=getattr(usage, 'total_completion_tokens', None),
		total_cost=getattr(usage, 'total_cost', None),
		steps=steps,
	)
