from __future__ import annotations

import numpy as np

from verl import DataProto
from verl.experimental.star_ppo.reward_allocators.code_turn_level import CodeTurnLevelRewardAllocator
from verl.experimental.star_ppo.tools.code_verifier import CodeVerifierTool
from verl.experimental.star_ppo.workflows.code_iterative_workflow import CodeIterativeWorkflowRunner
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord, WorkflowTrace


def test_code_verifier_standard_input_passes():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\nimport sys\nprint(sys.stdin.read().strip())\n</code>",
            "tests": '[{"input": "hello\\n", "output": "hello"}]',
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1
    assert result["runner_count"] == 1


def test_code_verifier_supports_call_based_tests():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\ndef add(a, b):\n    return a + b\n</code>",
            "tests": {"inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"], "fn_name": "add"},
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1
    assert result["runner_count"] == 1


def test_code_verifier_rearrange_string_special_judge_accepts_alternate_answer():
    verifier = CodeVerifierTool(timeout_seconds=2)
    problem = "Rearrange the characters of s to form a new string r that is not equal to s, or report impossible."
    result = verifier(
        {
            "problem": problem,
            "code": "<code>\nprint('YES')\nprint('oc')\n</code>",
            "tests": '[{"input": "1\\nco", "output": "YES\\noc"}]',
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1
    assert result["checker_type"] == "cf_rearrange_string"


def test_code_parser_requires_single_outer_tag():
    value, legal = CodeIterativeWorkflowRunner._parse_tagged_text("<pseudocode>read input</pseudocode>", "pseudocode")
    assert value == "read input"
    assert legal is True

    value, legal = CodeIterativeWorkflowRunner._parse_tagged_text("Plan:\n<pseudocode>x</pseudocode>", "pseudocode")
    assert value == "x"
    assert legal is False


def _thin(traj_id: str) -> DataProto:
    return DataProto.from_dict(
        tensors={},
        non_tensors={
            "traj_id": np.array([traj_id], dtype=object),
            "model_id": np.array(["m"], dtype=object),
            "query_id": np.array(["q"], dtype=object),
            "agent_id": np.array(["a"], dtype=object),
        },
    )


def _llm_record(node_id: str, turn: int, agent_id: str, format_reward: float = 0.0) -> WorkflowExecutionRecord:
    return WorkflowExecutionRecord(
        query_id="q",
        node_id=node_id,
        agent_id=agent_id,
        model_id="m",
        turn_id=turn,
        step_id=turn,
        node_type="llm",
        thin=_thin(f"{node_id}-traj"),
        meta={"format_reward": format_reward, "format_weight": 1.0},
    )


def _verifier_record(turn: int, pass_rate: float) -> WorkflowExecutionRecord:
    return WorkflowExecutionRecord(
        query_id="q",
        node_id=f"verifier_{turn}",
        agent_id="code_verifier",
        model_id="tool",
        turn_id=turn,
        step_id=turn,
        node_type="tool",
        trainable=False,
        parsed_output={"pass_rate": pass_rate, "all_passed": int(pass_rate == 1.0)},
    )


def test_code_reward_allocator_turn_deltas_and_format_penalty():
    allocator = CodeTurnLevelRewardAllocator(trainer=None, config=None)
    trace = WorkflowTrace(
        query_id="q",
        question="p",
        ground_truth=[],
        records=[
            _llm_record("planner_0", 0, "planner_agent"),
            _llm_record("coder_0", 0, "coder_agent", format_reward=-1.0),
            _verifier_record(0, 0.25),
            _llm_record("reflection_0", 0, "reflection_agent"),
            _llm_record("planner_1", 1, "planner_agent"),
            _llm_record("coder_1", 1, "coder_agent"),
            _verifier_record(1, 0.75),
        ],
    )

    assignments, metrics = allocator.allocate(trace)
    by_node = {assignment.record.node_id: assignment.reward for assignment in assignments}

    assert by_node["planner_0"] == 0.25
    assert by_node["coder_0"] == -0.75
    assert by_node["reflection_0"] == 0.5
    assert by_node["planner_1"] == 0.5
    assert by_node["coder_1"] == 0.5
    assert metrics["workflow/code/final_pass_rate"] == 0.75
