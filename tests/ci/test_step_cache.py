"""Unit tests for the agent step cache (memory layer for deterministic replay)."""

import pytest

from browser_use import Tools
from browser_use.agent.step_cache import (
	AgentStepCache,
	CachedElement,
	CachedStep,
	build_step_cache,
	classify_step,
	playwright_command,
)
from browser_use.agent.views import ActionResult, AgentHistory, AgentHistoryList, AgentOutput
from browser_use.browser.views import BrowserStateHistory
from browser_use.dom.views import DOMInteractedElement, NodeType


@pytest.fixture(scope='module')
def action_model():
	return Tools().registry.create_action_model()


@pytest.fixture(scope='module')
def agent_output_cls(action_model):
	return AgentOutput.type_with_custom_actions(action_model)


def _element(node_name='input', attributes=None, x_path='/html/body/div[1]', ax_name=None):
	return DOMInteractedElement(
		node_id=1,
		backend_node_id=1,
		frame_id=None,
		node_type=NodeType.ELEMENT_NODE,
		node_value='',
		node_name=node_name,
		attributes=attributes or {},
		bounds=None,
		x_path=x_path,
		element_hash=123,
		ax_name=ax_name,
	)


def _history_item(agent_output_cls, action_model, actions, results, url, elements):
	"""Build one AgentHistory item from (ActionModel kwargs, ActionResult, element) triples."""
	output = agent_output_cls(action=[action_model(**kwargs) for kwargs in actions])
	return AgentHistory(
		model_output=output,
		result=results,
		state=BrowserStateHistory(
			url=url,
			title='Example',
			tabs=[],
			interacted_element=elements,
			screenshot_path=None,
		),
		metadata=None,
	)


# ---- playwright_command ----


@pytest.mark.unit
def test_playwright_command_prefers_id():
	element = CachedElement(tag_name='input', attributes={'id': 'fullname', 'name': '01___title'})
	assert playwright_command(element) == 'locator("#fullname").first'


@pytest.mark.unit
def test_playwright_command_name_attribute_on_form_controls():
	element = CachedElement(tag_name='input', attributes={'name': '01___title'})
	assert playwright_command(element) == 'locator("[name=\'01___title\']").first'


@pytest.mark.unit
def test_playwright_command_href_on_anchor():
	element = CachedElement(tag_name='a', attributes={'href': '/ebooks/84'}, ax_name='Frankenstein 80366 downloads')
	assert playwright_command(element) == 'locator("a[href=\'/ebooks/84\']").first'


@pytest.mark.unit
def test_playwright_command_summary_uses_text_not_role():
	element = CachedElement(tag_name='summary', attributes={}, ax_name='Other formats & older devices')
	assert playwright_command(element) == 'get_by_text("Other formats & older devices")'


@pytest.mark.unit
def test_playwright_command_role_from_ax_name():
	element = CachedElement(tag_name='button', attributes={}, ax_name='Sign In')
	assert playwright_command(element) == 'get_by_role("button", name="Sign In")'


@pytest.mark.unit
def test_playwright_command_placeholder_fallback():
	element = CachedElement(tag_name='input', attributes={'placeholder': 'Enter email'})
	assert playwright_command(element) == 'get_by_placeholder("Enter email")'


@pytest.mark.unit
def test_playwright_command_xpath_last_resort():
	element = CachedElement(tag_name='div', attributes={}, x_path='/html/body/div[2]')
	assert playwright_command(element) == 'locator("xpath=/html/body/div[2]").first'


@pytest.mark.unit
def test_playwright_command_empty_when_nothing_available():
	element = CachedElement(tag_name='div', attributes={}, x_path=None, ax_name=None)
	assert playwright_command(element) == ''


# ---- classify_step ----


@pytest.mark.unit
def test_classify_done_is_terminal():
	classification, _ = classify_step('done', {}, None, success=True, is_done=True)
	assert classification == 'terminal'


@pytest.mark.unit
def test_classify_failed_step_is_redundant():
	classification, _ = classify_step('click', {'index': 5}, _element(), success=False, is_done=False)
	assert classification == 'redundant'


@pytest.mark.unit
def test_classify_scroll_is_redundant():
	classification, _ = classify_step('scroll', {'down': True}, None, success=True, is_done=False)
	assert classification == 'redundant'


@pytest.mark.unit
def test_classify_navigate_to_new_url_is_deterministic():
	classification, _ = classify_step(
		'navigate',
		{'url': 'https://example.com/other'},
		None,
		success=True,
		is_done=False,
		start_url='https://example.com',
	)
	assert classification == 'deterministic'


@pytest.mark.unit
def test_classify_navigate_to_start_url_is_redundant():
	classification, _ = classify_step(
		'navigate',
		{'url': 'https://example.com'},
		None,
		success=True,
		is_done=False,
		start_url='https://example.com',
	)
	assert classification == 'redundant'


@pytest.mark.unit
def test_classify_click_with_element_is_deterministic():
	classification, _ = classify_step('click', {'index': 5}, _element(), success=True, is_done=False)
	assert classification == 'deterministic'


@pytest.mark.unit
def test_classify_click_without_element_is_redundant():
	classification, _ = classify_step('click', {'coordinate_x': 1, 'coordinate_y': 2}, None, success=True, is_done=False)
	assert classification == 'redundant'


# ---- build_step_cache ----


@pytest.mark.unit
def test_build_step_cache_from_history(agent_output_cls, action_model, tmp_path):
	start_url = 'https://www.roboform.com/filling-test-all-fields'
	items = [
		# 1. type into the title field
		_history_item(
			agent_output_cls,
			action_model,
			[{'input': {'index': 6, 'text': 'myname'}}],
			[ActionResult()],
			start_url,
			[_element(attributes={'name': '01___title'})],
		),
		# 2. exploratory scroll (no state change needed for replay)
		_history_item(
			agent_output_cls,
			action_model,
			[{'scroll': {'down': True}}],
			[ActionResult()],
			start_url,
			[None],
		),
		# 3. failed click attempt (exploration)
		_history_item(
			agent_output_cls,
			action_model,
			[{'click': {'index': 42}}],
			[ActionResult(error='Element not found')],
			start_url,
			[_element(node_name='a', ax_name='wrong link')],
		),
		# 4. click the submit button
		_history_item(
			agent_output_cls,
			action_model,
			[{'click': {'index': 67}}],
			[ActionResult()],
			start_url,
			[_element(node_name='button', attributes={'id': 'submit-btn'})],
		),
		# 5. done
		_history_item(
			agent_output_cls,
			action_model,
			[{'done': {'text': 'Form filled', 'success': True}}],
			[ActionResult(is_done=True, success=True)],
			start_url,
			[None],
		),
	]
	history = AgentHistoryList(history=items)

	cache = build_step_cache(history, task='fill the form', start_url=start_url)

	assert cache.total_steps == 5
	assert cache.summary() == 'AgentStepCache: 5 steps total | 2 deterministic | 2 redundant | 1 terminal'

	steps = cache.steps
	assert steps[0].classification == 'deterministic'
	assert steps[0].action_name == 'input'
	assert steps[0].text == 'myname'
	assert steps[0].element.attributes == {'name': '01___title'}

	assert steps[1].classification == 'redundant'
	assert steps[2].classification == 'redundant'
	assert steps[2].error == 'Element not found'

	assert steps[3].classification == 'deterministic'
	assert steps[4].classification == 'terminal'

	# deterministic replay commands are sourced from the recorded fingerprints
	commands = cache.to_playwright_commands()
	assert [c['command'] for c in commands] == [
		'locator("[name=\'01___title\']").first',
		'locator("#submit-btn").first',
	]
	assert commands[0]['value'] == 'myname'

	# optexity automation conversion
	automation = cache.to_optexity_automation_dict()
	assert automation['url'] == start_url
	assert len(automation['nodes']) == 2
	first = automation['nodes'][0]['interaction_action']['input_text']
	assert first['command'] == 'locator("[name=\'01___title\']").first'
	assert first['input_text'] == 'myname'
	second = automation['nodes'][1]['interaction_action']['click_element']
	assert second['command'] == 'locator("#submit-btn").first'

	# persistence roundtrip
	cache_file = tmp_path / 'agent_step_cache.json'
	cache.save_to_file(cache_file)
	loaded = AgentStepCache.load_from_file(cache_file)
	assert loaded.model_dump() == cache.model_dump()


@pytest.mark.unit
def test_build_step_cache_navigate_to_new_page(agent_output_cls, action_model):
	start_url = 'https://example.com/home'
	items = [
		_history_item(
			agent_output_cls,
			action_model,
			[{'navigate': {'url': 'https://example.com/downloads'}}],
			[ActionResult()],
			start_url,
			[None],
		),
		_history_item(
			agent_output_cls,
			action_model,
			[{'navigate': {'url': start_url}}],
			[ActionResult()],
			'https://example.com/downloads',
			[None],
		),
	]
	cache = build_step_cache(AgentHistoryList(history=items), task='nav', start_url=start_url)

	# navigating away is deterministic, navigating back to the start URL is not
	assert [s.classification for s in cache.steps] == ['deterministic', 'redundant']

	automation = cache.to_optexity_automation_dict()
	assert len(automation['nodes']) == 1
	assert automation['nodes'][0]['interaction_action']['go_to_url']['url'] == 'https://example.com/downloads'
	assert automation['nodes'][0]['interaction_action']['go_to_url']['new_tab'] is False


@pytest.mark.unit
def test_replay_steps_dedupes_consecutive_duplicates(agent_output_cls, action_model):
	url = 'https://example.com/form'
	# browser-use typed the same text into the same field twice while exploring
	items = [
		_history_item(
			agent_output_cls,
			action_model,
			[{'input': {'index': 6, 'text': 'myname'}}],
			[ActionResult()],
			url,
			[_element(attributes={'name': '01___title'})],
		),
		_history_item(
			agent_output_cls,
			action_model,
			[{'input': {'index': 6, 'text': 'myname'}}],
			[ActionResult()],
			url,
			[_element(attributes={'name': '01___title'})],
		),
		# a different field is NOT a duplicate
		_history_item(
			agent_output_cls,
			action_model,
			[{'input': {'index': 7, 'text': 'myname'}}],
			[ActionResult()],
			url,
			[_element(attributes={'name': '02___address1'})],
		),
	]
	cache = build_step_cache(AgentHistoryList(history=items), task='dedup', start_url=url)

	# both inputs stay in the audit trail...
	assert len(cache.deterministic_steps()) == 3
	# ...but the replay collapses the consecutive duplicate
	assert len(cache.replay_steps()) == 2
	assert len(cache.to_optexity_automation_dict()['nodes']) == 2


@pytest.mark.unit
def test_send_keys_enter_folds_into_previous_input(agent_output_cls, action_model):
	url = 'https://example.com/search'
	items = [
		_history_item(
			agent_output_cls,
			action_model,
			[{'input': {'index': 6, 'text': 'my query'}}],
			[ActionResult()],
			url,
			[_element(node_name='input', attributes={'name': 'q'})],
		),
		_history_item(
			agent_output_cls,
			action_model,
			[{'send_keys': {'keys': 'Enter'}}],
			[ActionResult()],
			url,
			[None],
		),
	]
	cache = build_step_cache(AgentHistoryList(history=items), task='search', start_url=url)

	# the Enter keypress is folded into the input step, not replayed separately
	assert cache.steps[-1].classification == 'redundant'
	assert 'press_enter' in cache.steps[-1].reason
	assert cache.steps[0].press_enter is True

	nodes = cache.to_optexity_automation_dict()['nodes']
	assert len(nodes) == 1
	assert nodes[0]['interaction_action']['input_text']['press_enter'] is True


@pytest.mark.unit
def test_click_download_sets_expect_download(agent_output_cls, action_model):
	url = 'https://example.com/reports'
	items = [
		_history_item(
			agent_output_cls,
			action_model,
			[{'click': {'index': 12}}],
			[ActionResult(attachments=['quarterly-report.pdf'])],
			url,
			[_element(node_name='a', attributes={'id': 'download-btn'})],
		),
	]
	cache = build_step_cache(AgentHistoryList(history=items), task='download', start_url=url)

	assert cache.steps[0].downloaded_files == ['quarterly-report.pdf']

	node = cache.to_optexity_automation_dict()['nodes'][0]['interaction_action']['click_element']
	assert node['expect_download'] is True
	assert node['download_filename'] == 'quarterly-report.pdf'


@pytest.mark.unit
def test_cache_captures_run_metrics(agent_output_cls, action_model):
	from browser_use.tokens.views import UsageSummary

	usage = UsageSummary(
		total_prompt_tokens=1000,
		total_prompt_cost=0.01,
		total_prompt_cached_tokens=0,
		total_prompt_cached_cost=0.0,
		total_completion_tokens=200,
		total_completion_cost=0.002,
		total_tokens=1200,
		total_cost=0.012,
		entry_count=5,
	)
	items = [
		_history_item(
			agent_output_cls,
			action_model,
			[{'input': {'index': 6, 'text': 'myname'}}],
			[ActionResult()],
			'https://example.com',
			[_element(attributes={'name': '01___title'})],
		),
	]
	history = AgentHistoryList(history=items, usage=usage)

	cache = build_step_cache(history, task='metrics', start_url='https://example.com')

	assert cache.total_prompt_tokens == 1000
	assert cache.total_completion_tokens == 200
	assert cache.total_cost == 0.012
	assert cache.total_duration_seconds == 0.0


# ---- inline-plaintext download compilation ----


@pytest.mark.unit
def test_inline_plaintext_click_compiles_to_download_script():
	cache = AgentStepCache(
		task='download frankenstein',
		created_at='2026-01-01T00:00:00Z',
		steps=[
			CachedStep(
				step_index=0,
				url='https://www.gutenberg.org/ebooks/84',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': '/ebooks/84.txt.utf-8'}),
				classification='deterministic',
				reason='click on a concrete element',
			)
		],
	)
	nodes = cache.to_optexity_automation_dict()['nodes']
	assert len(nodes) == 1
	node = nodes[0]
	assert 'python_script_action' in node and 'interaction_action' not in node
	code = node['python_script_action']['execution_code']
	assert 'https://www.gutenberg.org/ebooks/84.txt.utf-8' in code
	assert 'ctx.save_download("84.txt.utf-8"' in code


@pytest.mark.unit
def test_download_script_uses_cached_absolute_url():
	cache = AgentStepCache(
		task='t',
		created_at='2026-01-01T00:00:00Z',
		steps=[
			CachedStep(
				step_index=0,
				url='https://example.com/books',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': 'https://cdn.example.com/readme.md'}),
				classification='deterministic',
				reason='click',
			)
		],
	)
	code = cache.to_optexity_automation_dict()['nodes'][0]['python_script_action']['execution_code']
	assert '"https://cdn.example.com/readme.md"' in code
	assert 'ctx.save_download("readme.md"' in code


@pytest.mark.unit
def test_inline_plaintext_click_prefers_cached_landing_url():
	"""A click whose href 302-redirects must fetch the cache's final landing URL:
	the in-page fetch() API refuses https->http redirect downgrades
	('TypeError: Failed to fetch'), breaking direct-href fetches for sites like
	Project Gutenberg. The landing URL is recorded in the cache's next step.
	"""
	cache = AgentStepCache(
		task='download frankenstein',
		created_at='2026-01-01T00:00:00Z',
		steps=[
			CachedStep(
				step_index=0,
				url='https://www.gutenberg.org/ebooks/84',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': '/ebooks/84.txt.utf-8'}),
				classification='deterministic',
				reason='click on a concrete element',
			),
			CachedStep(
				step_index=1,
				url='https://www.gutenberg.org/cache/epub/84/pg84.txt',
				action_name='done',
				classification='terminal',
				reason='task finished',
			),
		],
	)
	code = cache.to_optexity_automation_dict()['nodes'][0]['python_script_action']['execution_code']
	assert '"https://www.gutenberg.org/cache/epub/84/pg84.txt"' in code
	# filename follows the landing URL's basename, not the redirecting href
	assert 'ctx.save_download("pg84.txt"' in code
	assert 'ebooks/84.txt.utf-8' not in code


@pytest.mark.unit
def test_landing_url_ignored_when_not_plaintext():
	"""A differing next-step URL that is not plaintext must NOT be used as the
	fetch target (the click may have navigated to an HTML page)."""
	cache = AgentStepCache(
		task='t',
		created_at='2026-01-01T00:00:00Z',
		steps=[
			CachedStep(
				step_index=0,
				url='https://example.com/book',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': '/readme.txt'}),
				classification='deterministic',
				reason='click',
			),
			CachedStep(
				step_index=1,
				url='https://example.com/thanks',
				action_name='done',
				classification='terminal',
				reason='done',
			),
		],
	)
	code = cache.to_optexity_automation_dict()['nodes'][0]['python_script_action']['execution_code']
	assert '"https://example.com/readme.txt"' in code
	assert 'ctx.save_download("readme.txt"' in code


@pytest.mark.unit
def test_regular_clicks_stay_click_nodes():
	cache = AgentStepCache(
		task='t',
		created_at='2026-01-01T00:00:00Z',
		steps=[
			CachedStep(
				step_index=0,
				url='https://example.com',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': '/ebooks/84'}),
				classification='deterministic',
				reason='click',
			),
			CachedStep(
				step_index=1,
				url='https://example.com',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': '/pack.zip'}),
				classification='deterministic',
				reason='click',
			),
		],
	)
	nodes = cache.to_optexity_automation_dict()['nodes']
	assert all('interaction_action' in node for node in nodes)
	assert all(node['interaction_action'].get('click_element') for node in nodes)


@pytest.mark.unit
def test_click_with_recorded_download_stays_click():
	cache = AgentStepCache(
		task='t',
		created_at='2026-01-01T00:00:00Z',
		steps=[
			CachedStep(
				step_index=0,
				url='https://example.com',
				action_name='click',
				element=CachedElement(tag_name='a', attributes={'href': '/files/84/84.txt'}),
				downloaded_files=['84.txt'],
				classification='deterministic',
				reason='click downloaded a file',
			)
		],
	)
	node = cache.to_optexity_automation_dict()['nodes'][0]
	click = node['interaction_action']['click_element']
	assert click['expect_download'] is True
	assert click['download_filename'] == '84.txt'
