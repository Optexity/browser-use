import pytest

from browser_use.agent.views import (
	ActionResult,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	BrowserStateHistory,
	StepMetadata,
)
from browser_use.learning.step_cache import StepCache, build_optexity_automation


def _element(*, node_name: str, name: str, ax_name: str = '', element_hash: int) -> dict:
	return {
		'node_name': node_name,
		'attributes': {'name': name, 'type': 'text'},
		'ax_name': ax_name or name,
		'element_hash': element_hash,
		'x_path': f'/html/body/input[@name="{name}"]',
	}


def _history_item(actions: list[dict], elements: list[dict | None], results: list[ActionResult], next_goal: str):
	from browser_use.tools.registry.views import ActionModel

	action_models = [ActionModel(**a) for a in actions]
	model_output = AgentOutput(
		evaluation_previous_goal='success',
		memory='',
		next_goal=next_goal,
		action=action_models,
	)
	return AgentHistory(
		model_output=model_output,
		result=results,
		state=BrowserStateHistory(
			url='https://example.com/form',
			title='Form',
			tabs=[],
			interacted_element=elements,
		),
		metadata=StepMetadata(step_number=1, step_start_time=0, step_end_time=1),
	)


def test_step_cache_keeps_inputs_and_drops_exploration():
	history = AgentHistoryList(history=[])
	history.history = [
		_history_item(
			[{'scroll': {'down': True, 'num_pages': 1.0}}],
			[None],
			[ActionResult(extracted_content='scrolled')],
			'scroll to form',
		),
		_history_item(
			[{'input': {'index': 1, 'text': 'myname', 'clear': True}}],
			[_element(node_name='input', name='fullname', element_hash=101)],
			[ActionResult(extracted_content="Typed 'myname'", long_term_memory="Typed 'myname'")],
			'fill full name',
		),
		_history_item(
			[{'input': {'index': 2, 'text': 'xyz', 'clear': True}}],
			[_element(node_name='input', name='address1', element_hash=102)],
			[ActionResult(extracted_content="Typed 'xyz'", long_term_memory="Typed 'xyz'")],
			'fill address line 1',
		),
		_history_item(
			[{'done': {'text': 'done', 'success': True}}],
			[None],
			[ActionResult(extracted_content='done', is_done=True, success=True)],
			'finish',
		),
	]

	cache = StepCache(start_url='https://example.com/form', task='fill form').process_history(history)

	assert len(cache.deterministic_steps) == 2
	assert cache.deterministic_steps[0].input_text == 'myname'
	assert cache.deterministic_steps[0].command == 'locator("input[name=\'fullname\']").first'
	assert cache.deterministic_steps[1].input_text == 'xyz'
	assert any(s.action_type == 'scroll' for s in cache.skipped_steps)


def test_build_optexity_automation_from_cache():
	from browser_use.learning.step_cache import CachedStep

	cache = StepCache(start_url='https://example.com/form')
	cache.deterministic_steps = [
		CachedStep(
			action_type='input_text',
			command='locator("input[name=\'city\']").first',
			prompt_instructions='fill city',
			input_text='SF',
		)
	]

	automation = build_optexity_automation(cache)
	assert automation['url'] == 'https://example.com/form'
	assert len(automation['nodes']) == 1
	node = automation['nodes'][0]['interaction_action']['input_text']
	assert node['command'] == 'locator("input[name=\'city\']").first'
	assert node['input_text'] == 'SF'
	assert node['skip_prompt'] is True
