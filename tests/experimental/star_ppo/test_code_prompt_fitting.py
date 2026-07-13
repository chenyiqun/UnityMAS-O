# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from verl.experimental.star_ppo.workflows.code_iterative_workflow import CodeIterativeWorkflowRunner


class _CharacterTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in str(text)]

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        del kwargs
        text = "".join(str(message.get("content", "")) for message in messages)
        if add_generation_prompt:
            text += "<>"
        return self.encode(text) if tokenize else text


def _runner(prompt_limit: int) -> CodeIterativeWorkflowRunner:
    runner = object.__new__(CodeIterativeWorkflowRunner)
    runner.trainer = SimpleNamespace(tokenizer=_CharacterTokenizer())
    runner.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout={"prompt_length": prompt_limit}),
        data={},
    )
    runner.per_infer_prompt_max_tokens = prompt_limit
    runner.code_cfg = {"max_state_chars": 12000}
    runner.max_turns = 3
    return runner


def test_code_context_fitting_preserves_task_and_output_contract():
    runner = _runner(prompt_limit=900)
    template = (
        "ROLE\nProblem:\n{original_problem}\nTests:\n{visible_tests}\nState:\n{global_state_text}\n"
        "Current code:\n{current_code}\nError:\n{current_error}\n"
        "Task: fix the program.\nOutput only:\n<code>source</code>"
    )
    context = {
        "original_problem": "problem " * 400,
        "visible_tests": "test " * 250,
        "global_state_text": "history " * 300,
        "starter_code": "",
        "current_code": "print(1)\n" * 120,
        "current_pseudocode": "",
        "current_error": "wrong answer " * 80,
    }

    fitted, trimmed_tokens = runner._prepare_prompt_context(
        "reflection_0",
        {"prompt_template": template},
        context,
    )
    rendered = runner._render_template(template, fitted)

    assert trimmed_tokens > 0
    assert runner._count_chat_tokens([{"role": "user", "content": rendered}]) <= 900
    assert rendered.endswith("Task: fix the program.\nOutput only:\n<code>source</code>")


def test_fallback_prompt_truncation_preserves_head_and_output_suffix():
    runner = _runner(prompt_limit=160)
    prompt = "ROLE AND INSTRUCTIONS\n" + ("large input " * 100) + "\nOUTPUT CONTRACT <code>source</code>"

    messages, trimmed_tokens, prompt_tokens = runner._build_truncated_chat_prompt(prompt)
    fitted = messages[0]["content"]

    assert trimmed_tokens > 0
    assert prompt_tokens <= 160
    assert fitted.startswith("ROLE AND INSTRUCTIONS")
    assert fitted.endswith("OUTPUT CONTRACT <code>source</code>")


def test_rendered_global_state_omits_static_fields_already_present_in_prompt():
    runner = _runner(prompt_limit=900)
    state = {
        "problem": "duplicated problem",
        "starter_code": "def solve(): pass",
        "fn_name": "solve",
        "execution_instruction": "duplicated execution instruction",
        "visible_tests": "duplicated tests",
        "iterations": [{"turn": 0, "reflection": "fix edge case"}],
    }

    rendered = runner._format_global_state(state)

    assert "duplicated problem" not in rendered
    assert "duplicated execution instruction" not in rendered
    assert "duplicated tests" not in rendered
    assert "def solve(): pass" in rendered
    assert "fix edge case" in rendered


def test_reflection_context_does_not_duplicate_current_iteration_in_global_state():
    runner = _runner(prompt_limit=900)
    state = {
        "problem": "problem",
        "starter_code": "",
        "fn_name": "",
        "execution_instruction": "stdio",
        "visible_tests": "test",
        "iterations": [
            {"turn": 0, "code": "old code", "reflection": "old reflection"},
            {"turn": 1, "code": "current code", "verification": {"pass_rate": 0.0}},
        ],
    }

    context = runner._build_prompt_context(
        problem="problem",
        state=state,
        turn_id=1,
        current_pseudocode="current plan",
        current_code="current code",
        current_error="wrong answer",
        exclude_current_iteration_from_state=True,
    )

    assert "old reflection" in context["global_state_text"]
    assert "current code" not in context["global_state_text"]
    assert context["current_code"] == "current code"


def test_coder_context_does_not_duplicate_explicit_starter_code_in_global_state():
    runner = _runner(prompt_limit=900)
    state = {
        "problem": "problem",
        "starter_code": "unique starter code",
        "fn_name": "",
        "execution_instruction": "stdio",
        "visible_tests": "test",
        "iterations": [],
    }

    context = runner._build_prompt_context(
        problem="problem",
        state=state,
        turn_id=0,
        current_pseudocode="plan",
        include_starter_code_in_state=False,
    )

    assert context["starter_code"] == "unique starter code"
    assert "unique starter code" not in context["global_state_text"]
