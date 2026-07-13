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

import asyncio

import pytest

from verl.experimental.star_ppo.reward_allocators.mask_turn_level import MAskTurnLevelRewardAllocator
from verl.experimental.star_ppo.tools.prompt_builders import MAskIterativeContextBuilder
from verl.experimental.star_ppo.workflows.mask_iterative_workflow import MAskIterativeWorkflowRunner
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord


def _knowledge_state(answer: str = "London") -> dict:
    return {
        "question": "What is the capital of France?",
        "thinking_trajectory": [
            {
                "step_id": "tau1",
                "sub_question": "What is the capital of France?",
                "sub_answer": answer,
            }
        ],
        "predicted_answer": answer,
    }


def test_context_builder_reads_summary_nodes_written_by_workflow():
    nodes = {
        "plan": {"knowledge_state": _knowledge_state()},
        "search_1": {"search_decision": {"action": "search", "query": "France capital"}},
        "summary_1": {
            "evidence_summary": {
                "query": "France capital",
                "summary": "Paris is the capital of France.",
                "salient_facts": [],
            }
        },
    }

    context = MAskIterativeContextBuilder()(nodes)

    assert "Turn 1: Query: France capital" in context["search_history_text"]
    assert "Turn 1: Summary: Paris is the capital of France." in context["search_history_text"]
    assert "(No summary yet)" not in context["search_history_text"]


@pytest.mark.parametrize(
    "response",
    [
        "<q1>first</q1><final_answer>answer</final_answer>",
        "<q2>second</q2><a2>answer</a2><final_answer>answer</final_answer>",
        "<q1>first</q1><a1></a1><final_answer>answer</final_answer>",
        "<q1>first</q1><a1>answer</a1><final_answer>x</final_answer><final_answer>y</final_answer>",
    ],
)
def test_plan_parser_rejects_incomplete_or_non_consecutive_pairs(response):
    _, is_legal = MAskIterativeWorkflowRunner._parse_plan_output(response, question="question")

    assert not is_legal


def test_plan_parser_accepts_complete_ordered_pairs():
    state, is_legal = MAskIterativeWorkflowRunner._parse_plan_output(
        "<q1>first</q1><a1>one</a1><q2>second</q2><a2>two</a2>"
        "<final_answer>two</final_answer>",
        question="question",
    )

    assert is_legal
    assert [item["step_id"] for item in state["thinking_trajectory"]] == ["tau1", "tau2"]
    assert state["predicted_answer"] == "two"


def test_empty_or_malformed_end_search_never_reaches_retriever():
    empty_action, empty_legal = MAskIterativeWorkflowRunner._parse_search_output("<search></search>")
    malformed_end_action, malformed_end_legal = MAskIterativeWorkflowRunner._parse_search_output("<end> extra")

    assert empty_action == {"action": "end", "query": ""}
    assert not empty_legal
    assert malformed_end_action == {"action": "end", "query": ""}
    assert not malformed_end_legal


def test_invalid_update_target_falls_back_to_an_add_operation():
    runner = object.__new__(MAskIterativeWorkflowRunner)

    parsed, is_legal = runner._parse_update_output(
        "<update>t99</update>",
        prev_state=_knowledge_state(),
        query="France capital",
        evidence="Paris is the capital of France.",
    )

    assert not is_legal
    assert parsed["operation"] == "add"
    assert parsed["target_step"] == "t2"
    assert len(parsed["knowledge_state"]["thinking_trajectory"]) == 2


def test_full_search_turn_updates_state_stops_and_allocates_rewards():
    runner = object.__new__(MAskIterativeWorkflowRunner)
    runner.max_turns = 2
    runner.stop_on_search_end = True
    runner.context_builder = MAskIterativeContextBuilder()
    # The shared-parameter configuration routes every trainable role to one
    # engine. agent_id must still retain each role's identity for reward and
    # per-agent metrics.
    shared_model_id = "shared_agent_llm"
    runner.plan_cfg = {"model_id": shared_model_id, "agent_id": "planning_agent"}
    runner.search_cfg = {"model_id": shared_model_id, "agent_id": "search_agent"}
    runner.summary_cfg = {"model_id": shared_model_id, "agent_id": "summary_agent"}
    runner.update_cfg = {"model_id": shared_model_id, "agent_id": "update_agent"}
    runner.answer_cfg = {"model_id": shared_model_id, "agent_id": "answer_agent"}
    runner.retriever_cfg = {"tool": "retriever", "top_k": 5, "max_attempts": 2, "fail_open": True}
    runner.debug_max_chars = 1000
    runner._extract_question = lambda _: "What is the capital of France?"
    runner._extract_gt_list = lambda _: ["Paris"]
    runner._extract_from_batch = lambda _, key: "query-1" if key == "query_id" else None

    responses = {
        "plan": (
            "<q1>What is the capital of France?</q1><a1>London</a1>"
            "<final_answer>London</final_answer>"
        ),
        "answer_0": "<final_answer>London</final_answer>",
        "search_1": "<search>France capital</search>",
        "summary_1": "<evidence>Paris is the capital of France.</evidence>",
        "update_1": "<update>t1</update>",
        "answer_1": "<final_answer>Paris</final_answer>",
        "search_2": "<end>",
    }
    seen_contexts = {}

    async def fake_execute_llm_step(**kwargs):
        node_id = kwargs["node_id"]
        node_cfg = kwargs["node_cfg"]
        seen_contexts[node_id] = kwargs["prompt_context"]
        return WorkflowExecutionRecord(
            query_id="query-1",
            node_id=node_id,
            agent_id=node_cfg["agent_id"],
            model_id=node_cfg["model_id"],
            turn_id=kwargs["turn_id"],
            step_id=kwargs["step_id"],
            node_type="llm",
            raw_output=responses[node_id],
            meta={"format_weight": 1.0},
        )

    async def fake_run_tool_record(**kwargs):
        return WorkflowExecutionRecord(
            query_id="query-1",
            node_id=kwargs["node_id"],
            agent_id=kwargs["tool_name"],
            model_id="tool",
            turn_id=kwargs["turn_id"],
            step_id=kwargs["step_id"],
            node_type="tool",
            raw_output="Paris is the capital of France.",
            parsed_output=["Paris is the capital of France."],
            trainable=False,
            state_before=kwargs["state_before"],
            state_after=kwargs["state_after"],
        )

    runner._execute_llm_step = fake_execute_llm_step
    runner._run_tool_record = fake_run_tool_record

    trace = asyncio.run(runner.run_query(object(), query_local_idx=0, debug=False))

    assert [record.node_id for record in trace.records] == [
        "plan",
        "answer_0",
        "search_1",
        "retrieve_1",
        "summary_1",
        "update_1",
        "answer_1",
        "search_2",
    ]
    assert trace.state["predicted_answer"] == "Paris"
    assert trace.metrics["workflow/mask/stopped_by_search_end"] == 1.0
    assert seen_contexts["update_1"]["evidence_summary"] == "Paris is the capital of France."
    assert "Turn 1: Summary: Paris is the capital of France." in seen_contexts["search_2"]["search_history_text"]
    trainable_records = [record for record in trace.records if record.trainable]
    assert {record.model_id for record in trainable_records} == {shared_model_id}
    assert {record.agent_id for record in trainable_records} == {
        "planning_agent",
        "search_agent",
        "summary_agent",
        "update_agent",
        "answer_agent",
    }

    allocator = object.__new__(MAskTurnLevelRewardAllocator)
    assignments, metrics = allocator.allocate(trace)
    assignments_by_node = {assignment.record.node_id: assignment for assignment in assignments}

    assert assignments_by_node["search_1"].meta["task_reward"] == 1.0
    assert assignments_by_node["summary_1"].meta["task_reward"] == 1.0
    assert assignments_by_node["update_1"].meta["task_reward"] == 1.0
    assert assignments_by_node["answer_1"].meta["task_reward"] == 1.0
    assert assignments_by_node["search_2"].meta["task_reward"] == 0.0
    assert metrics["workflow/mask/final_f1"] == 1.0
