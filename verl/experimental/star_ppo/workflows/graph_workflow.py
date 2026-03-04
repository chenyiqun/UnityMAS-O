from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

import numpy as np

from verl import DataProto
from verl.experimental.star_ppo.tools import build_retriever_tool
from verl.experimental.star_ppo.workflows.base import WorkflowRunner
from verl.utils.import_utils import load_extern_object
from verl.utils.reward_score.search_r1_like_qa_em import em_check


class GraphWorkflowRunner(WorkflowRunner):
    """Configurable query-level workflow graph runner.

    Graph format (under `star.workflow.graph`):
    - `start_nodes`: list[str]
    - `end_nodes`: list[str]
    - `max_steps`: int
    - `nodes`: dict[node_id, node_cfg]
      - llm node:
        - `type: llm`
        - `model_id`, `agent_id`, `prompt_template`
        - `parser`: {`type`: `json_key|raw`, `key`: str}
        - `output_key`: str
        - `reward`: {`format_weight`: float, `share_outcome`: bool}
        - `next`: list[str|{to: str, when: str}]
      - tool node:
        - `type: tool`
        - `tool`: tool alias in `star.workflow.tools`
        - `input_template`: str
        - `top_k`: int (for retriever-like tools)
        - `output_key`: str
        - `next`: list[str|{to: str, when: str}]
    - `outcome_reward`: {`type`: `em`, `source`: "node_id.output_key", `weight`: float}
    """

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        self.workflow_cfg = self.config.star.get("workflow", {})
        self.graph_cfg = self.workflow_cfg.get("graph", {})
        self.nodes = dict(self.graph_cfg.get("nodes", {}))
        self.start_nodes = list(self.graph_cfg.get("start_nodes", []))
        self.end_nodes = set(self.graph_cfg.get("end_nodes", []))
        self.max_steps = int(self.graph_cfg.get("max_steps", 16))
        self.stop_on_end = bool(self.graph_cfg.get("stop_on_end", True))
        self.max_inflight_queries = int(self.workflow_cfg.get("max_inflight_queries", 32))
        self.question_candidates = list(self.workflow_cfg.get("question_candidates", ["question", "query", "problem"]))
        self.gt_candidates = list(
            self.workflow_cfg.get(
                "ground_truth_candidates",
                ["ground_truth", "answer", "target", "golden_answers", "reward_model"],
            )
        )
        self.outcome_cfg = self.graph_cfg.get("outcome_reward", {"type": "em", "source": "", "weight": 1.0})
        self.tools = self._build_tools()
        self._validate_graph()

    def _build_tools(self) -> dict[str, Any]:
        tools = {}
        for alias, tool_cfg in dict(self.workflow_cfg.get("tools", {})).items():
            # Built-in retriever adapters.
            if str(tool_cfg.get("type", "")) in {"simple_keyword", "http"}:
                tools[alias] = build_retriever_tool(tool_cfg)
                continue

            # External plugin tool object.
            if "path" in tool_cfg and "name" in tool_cfg:
                obj = load_extern_object(str(tool_cfg.get("path")), str(tool_cfg.get("name")))
                kwargs = dict(tool_cfg.get("kwargs", {}))
                tools[alias] = obj(**kwargs) if isinstance(obj, type) else obj
        return tools

    def _validate_graph(self):
        if len(self.start_nodes) == 0:
            raise ValueError("star.workflow.graph.start_nodes must be non-empty")
        for node_id in self.start_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"start node not found: {node_id}")
        for node_id in self.end_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"end node not found: {node_id}")

        for node_id, cfg in self.nodes.items():
            node_type = str(cfg.get("type", "llm"))
            if node_type not in {"llm", "tool"}:
                raise ValueError(f"node {node_id} has unsupported type={node_type}")
            if node_type == "llm":
                model_id = str(cfg.get("model_id", ""))
                if model_id not in self.trainer.model_ids:
                    raise ValueError(
                        f"node {node_id} uses unknown model_id={model_id}, available={self.trainer.model_ids}"
                    )
            if node_type == "tool":
                tool_alias = str(cfg.get("tool", ""))
                if tool_alias not in self.tools:
                    raise ValueError(
                        f"node {node_id} uses unknown tool alias={tool_alias}, "
                        f"available={sorted(self.tools.keys())}"
                    )
            for edge in self._normalize_edges(cfg.get("next", [])):
                to_node = edge["to"]
                if to_node not in self.nodes:
                    raise ValueError(f"node {node_id} has edge to missing node {to_node}")

    @staticmethod
    def _normalize_edges(edges_cfg) -> list[dict[str, Any]]:
        edges = []
        for item in list(edges_cfg):
            if isinstance(item, str):
                edges.append({"to": item, "when": None})
            elif isinstance(item, dict) and "to" in item:
                edges.append({"to": str(item["to"]), "when": item.get("when", None)})
        return edges

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    @staticmethod
    def _as_template_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list | tuple):
            return "\n".join([str(x) for x in v])
        return str(v)

    def _lookup_path(self, context: dict[str, Any], dotted: str, default: Any = "") -> Any:
        cur: Any = context
        for key in str(dotted).split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    def _render_template(self, template: str, context: dict[str, Any]) -> str:
        # Respect escaped braces while supporting dotted-path placeholders.
        s = str(template).replace("{{", "\0L").replace("}}", "\0R")

        def repl(match):
            key = match.group(1).strip()
            value = self._lookup_path(context, key, default="")
            return self._as_template_value(value)

        s = re.sub(r"\{([a-zA-Z0-9_.]+)\}", repl, s)
        return s.replace("\0L", "{").replace("\0R", "}")

    def _eval_when(self, when_expr: str | None, context: dict[str, Any]) -> bool:
        if when_expr is None or str(when_expr).strip() == "":
            return True
        try:
            env = {
                "step": context.get("step", 0),
                "question": context.get("question", ""),
                "ground_truth": context.get("ground_truth", []),
                "nodes": context.get("nodes", {}),
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "any": any,
                "all": all,
                "min": min,
                "max": max,
            }
            return bool(eval(str(when_expr), {"__builtins__": {}}, env))
        except Exception:
            return False

    def _parse_llm_output(self, raw_text: str, node_cfg) -> tuple[Any, float]:
        parser = node_cfg.get("parser", {})
        parser_type = str(parser.get("type", "raw"))
        output_key = str(node_cfg.get("output_key", parser.get("key", "output")))
        if parser_type == "json_key":
            obj = self.trainer._extract_first_json_object(raw_text)
            if isinstance(obj, dict):
                value = str(obj.get(str(parser.get("key", output_key)), "")).strip()
                return value, 1.0 if value else 0.0
            return raw_text.strip(), 0.0
        value = raw_text.strip()
        return value, 1.0 if value else 0.0

    def _extract_question(self, query_batch: DataProto) -> str:
        vec = self.trainer._extract_string_vector(query_batch, self.question_candidates, default="")
        return str(vec[0]) if len(vec) > 0 else ""

    def _extract_gt_list(self, query_batch: DataProto) -> list[str]:
        gts = self.trainer._extract_ground_truth_lists(query_batch, self.gt_candidates)
        return gts[0] if len(gts) > 0 else []

    async def _execute_node(self, node_id: str, query_batch: DataProto, context: dict[str, Any]) -> dict[str, Any]:
        node_cfg = self.nodes[node_id]
        node_type = str(node_cfg.get("type", "llm"))
        if node_type == "llm":
            model_id = str(node_cfg["model_id"])
            agent_id = str(node_cfg.get("agent_id", node_id))
            prompt_text = self._render_template(str(node_cfg.get("prompt_template", "{question}")), context)
            prompt_batch = self.trainer._build_workflow_prompt_batch(
                query_batch,
                [[{"role": "user", "content": prompt_text}]],
                agent_id,
            )
            _, thin, _ = await self.trainer._rollout_model_async(model_id, prompt_batch)
            raw_text = str(thin.non_tensor_batch["action_text"][0])
            parsed_value, format_reward = self._parse_llm_output(raw_text, node_cfg)
            output_key = str(node_cfg.get("output_key", "output"))
            context["nodes"][node_id] = {
                "raw_text": raw_text,
                output_key: parsed_value,
                "format_reward": float(format_reward),
            }
            context[node_id] = context["nodes"][node_id]
            return {
                "node_id": node_id,
                "node_type": "llm",
                "thin": thin,
                "format_reward": float(format_reward),
                "output_key": output_key,
                "output_value": parsed_value,
            }

        if node_type == "tool":
            tool_name = str(node_cfg["tool"])
            if tool_name not in self.tools:
                raise ValueError(f"tool node {node_id} references missing tool alias={tool_name}")
            tool = self.tools[tool_name]
            input_text = self._render_template(str(node_cfg.get("input_template", "{question}")), context)
            if hasattr(tool, "retrieve"):
                top_k = int(node_cfg.get("top_k", 3))
                output = await asyncio.to_thread(tool.retrieve, input_text, top_k)
            elif callable(tool):
                output = await asyncio.to_thread(tool, input_text)
            else:
                raise TypeError(f"tool {tool_name} is not callable and has no retrieve()")
            output_key = str(node_cfg.get("output_key", "output"))
            context["nodes"][node_id] = {
                "input": input_text,
                output_key: output,
            }
            context[node_id] = context["nodes"][node_id]
            return {
                "node_id": node_id,
                "node_type": "tool",
                "output_key": output_key,
                "output_value": output,
            }

        raise ValueError(f"Unsupported node type for {node_id}: {node_type}")

    def _compute_outcome_reward(self, context: dict[str, Any]) -> float:
        outcome_type = str(self.outcome_cfg.get("type", "em"))
        if outcome_type != "em":
            return 0.0
        source = str(self.outcome_cfg.get("source", ""))
        weight = float(self.outcome_cfg.get("weight", 1.0))
        pred = str(self._lookup_path(context, source, default=""))
        gt = context.get("ground_truth", [])
        return weight * float(em_check(pred, gt)) if len(gt) > 0 else 0.0

    async def _run_one_query(self, query_batch: DataProto, query_sem: asyncio.Semaphore) -> dict[str, Any]:
        async with query_sem:
            context = {
                "question": self._extract_question(query_batch),
                "ground_truth": self._extract_gt_list(query_batch),
                "nodes": {},
                "step": 0,
            }
            frontier = list(self.start_nodes)
            llm_exec_records: list[dict[str, Any]] = []

            for step in range(self.max_steps):
                if len(frontier) == 0:
                    break
                context["step"] = step
                node_tasks = [self._execute_node(node_id, query_batch, context) for node_id in frontier]
                node_results = await asyncio.gather(*node_tasks)

                next_frontier: list[str] = []
                hit_end = False
                for result in node_results:
                    node_id = result["node_id"]
                    if result["node_type"] == "llm":
                        llm_exec_records.append(result)
                    if node_id in self.end_nodes:
                        hit_end = True
                    for edge in self._normalize_edges(self.nodes[node_id].get("next", [])):
                        if self._eval_when(edge["when"], context):
                            next_frontier.append(edge["to"])

                if self.stop_on_end and hit_end:
                    frontier = []
                else:
                    frontier = self._dedupe_keep_order(next_frontier)

            outcome_reward = self._compute_outcome_reward(context)
            reward_parts = []
            node_format = {}
            for rec in llm_exec_records:
                node_id = rec["node_id"]
                reward_cfg = dict(self.nodes[node_id].get("reward", {}))
                format_weight = float(reward_cfg.get("format_weight", 0.0))
                share_outcome = bool(reward_cfg.get("share_outcome", True))
                total = format_weight * float(rec["format_reward"]) + (outcome_reward if share_outcome else 0.0)
                reward_parts.append(self.trainer._build_commit_rewards_from_thin(rec["thin"], np.array([total], dtype=np.float32)))
                node_format[node_id] = float(rec["format_reward"])

            return {
                "reward_parts": reward_parts,
                "outcome_reward": float(outcome_reward),
                "node_format": node_format,
                "llm_node_count": float(len(llm_exec_records)),
            }

    async def run_batch(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        del epoch
        query_sem = asyncio.Semaphore(max(1, self.max_inflight_queries))
        tasks = [self._run_one_query(batch.select_idxs([i]), query_sem) for i in range(len(batch))]
        query_results = await asyncio.gather(*tasks)

        reward_parts = []
        outcome_rewards = []
        llm_node_counts = []
        node_format_acc = defaultdict(list)
        for item in query_results:
            reward_parts.extend(item["reward_parts"])
            outcome_rewards.append(item["outcome_reward"])
            llm_node_counts.append(item["llm_node_count"])
            for node_id, val in item["node_format"].items():
                node_format_acc[node_id].append(float(val))

        if len(reward_parts) == 0:
            rewards = self.trainer._empty_rewards()
        else:
            rewards = DataProto.concat(reward_parts) if len(reward_parts) > 1 else reward_parts[0]

        metrics = {
            "workflow/samples": float(len(query_results)),
            "workflow/outcome_reward_mean": float(np.mean(outcome_rewards)) if outcome_rewards else 0.0,
            "workflow/llm_nodes_per_query_mean": float(np.mean(llm_node_counts)) if llm_node_counts else 0.0,
        }
        for node_id, values in node_format_acc.items():
            metrics[f"workflow/node/{node_id}/format_reward_mean"] = float(np.mean(values)) if values else 0.0
        return rewards, metrics
