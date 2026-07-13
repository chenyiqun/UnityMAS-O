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

from verl.experimental.star_ppo.workflows.code_iterative_workflow import CodeIterativeWorkflowRunner
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord


def _record(raw_output: str, output_tokens: int, max_response_tokens: int = 8) -> WorkflowExecutionRecord:
    return WorkflowExecutionRecord(
        query_id="query",
        node_id="coder_0",
        agent_id="coder_agent",
        model_id="coder_llm",
        turn_id=0,
        step_id=0,
        node_type="llm",
        raw_output=raw_output,
        meta={
            "output_tokens": output_tokens,
            "max_response_tokens": max_response_tokens,
        },
    )


def _score(record: WorkflowExecutionRecord, tag: str = "code") -> bool:
    _, is_legal = CodeIterativeWorkflowRunner._parse_record_tagged_text(record, tag)
    CodeIterativeWorkflowRunner._set_record_format_reward(record, is_legal, tag)
    return is_legal


def test_old_format_semantics_allow_comments_inside_valid_code_tag():
    record = _record("<code># explanation\nprint(1)</code>", output_tokens=6)

    assert _score(record)
    assert record.meta["format_reward"] == 0.0
    assert record.meta["format_penalty_exempt"] is False


def test_unclosed_tag_at_response_limit_is_not_penalized_but_remains_illegal():
    record = _record("<code>print(1)", output_tokens=8)

    parsed, is_legal = CodeIterativeWorkflowRunner._parse_record_tagged_text(record, "code")
    CodeIterativeWorkflowRunner._set_record_format_reward(record, is_legal, "code")

    assert parsed == "print(1)"
    assert not is_legal
    assert record.meta["format_reward"] == 0.0
    assert record.meta["is_legal_format"] is False
    assert record.meta["hit_response_token_limit"] is True
    assert record.meta["format_penalty_exempt"] is True
    assert record.meta["format_penalty_exempt_reason"] == "unclosed_tag_at_response_token_limit"


def test_unclosed_tag_before_response_limit_is_penalized():
    record = _record("<code>print(1)", output_tokens=7)

    assert not _score(record)
    assert record.meta["format_reward"] == -1.0
    assert record.meta["format_penalty_exempt"] is False


def test_other_malformed_output_at_response_limit_is_still_penalized():
    record = _record("explanation\n<code>print(1)", output_tokens=8)

    assert not _score(record)
    assert record.meta["format_reward"] == -1.0
    assert record.meta["format_penalty_exempt"] is False


def test_duplicate_opening_tag_at_response_limit_is_still_penalized():
    record = _record("<code>print('<code>')", output_tokens=8)

    assert not _score(record)
    assert record.meta["format_reward"] == -1.0
    assert record.meta["format_penalty_exempt"] is False


def test_wrong_closing_tag_at_response_limit_is_still_penalized():
    record = _record("<code>print(1)</pseudocode>", output_tokens=8)

    assert not _score(record)
    assert record.meta["format_reward"] == -1.0
    assert record.meta["format_penalty_exempt"] is False
