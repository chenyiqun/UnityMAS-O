from __future__ import annotations

import asyncio
import os
import re
import string
from collections.abc import Mapping
from collections import Counter, defaultdict
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
        self.question_candidates = list(
            self.workflow_cfg.get("question_candidates", ["question", "query", "problem", "extra_info.question"])
        )
        self.gt_candidates = list(
            self.workflow_cfg.get(
                "ground_truth_candidates",
                [
                    "ground_truth",
                    "answer",
                    "target",
                    "golden_answers",
                    "extra_info.answer",
                    "reward_model.ground_truth",
                    "reward_model",
                ],
            )
        )
        self.outcome_cfg = self.graph_cfg.get("outcome_reward", {"type": "em", "source": "", "weight": 1.0})
        self.tools = self._build_tools()
        self._validate_graph()
        debug_cfg = dict(self.workflow_cfg.get("debug", {}))
        env_debug = str(os.environ.get("STAR_WORKFLOW_DEBUG", "")).strip().lower()
        self.debug_enabled = bool(debug_cfg.get("enabled", False)) or env_debug in {"1", "true", "yes", "on"}
        self.debug_sample_index = int(
            debug_cfg.get("sample_index", os.environ.get("STAR_WORKFLOW_DEBUG_SAMPLE_INDEX", 0))
        )
        self.debug_max_chars = int(debug_cfg.get("max_chars", os.environ.get("STAR_WORKFLOW_DEBUG_MAX_CHARS", 160)))
        self.debug_every_n_batches = max(
            1, int(debug_cfg.get("every_n_batches", os.environ.get("STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES", 20)))
        )
        self._debug_batch_counter = 0

    def _build_tools(self) -> dict[str, Any]:
        tools = {}
        for alias, tool_cfg in dict(self.workflow_cfg.get("tools", {})).items():
            # Built-in retriever adapters.
            if str(tool_cfg.get("type", "")) in {"simple_keyword", "http", "query_api_pool", "retrieval_api_pool"}:
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
            # OmegaConf DictConfig is Mapping-like but not a plain dict.
            elif isinstance(item, Mapping) and "to" in item:
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

    def _clip_debug_text(self, value: Any) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.debug_max_chars:
            return text
        return text[: max(0, self.debug_max_chars - 3)] + "..."

    def _summarize_debug_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            if len(value) == 0:
                return "list(len=0)"
            first = self._clip_debug_text(value[0])
            return f"list(len={len(value)}, first={first})"
        if isinstance(value, dict):
            keys = list(value.keys())
            preview = []
            for k in keys[:3]:
                v = value[k]
                if isinstance(v, str | int | float | bool):
                    preview.append(f"{k}={self._clip_debug_text(v)}")
            if preview:
                return f"dict(keys={keys[:5]}, {', '.join(preview)})"
            return f"dict(keys={keys[:5]})"
        return self._clip_debug_text(value)

    @staticmethod
    def _as_template_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list | tuple):
            return "\n".join([GraphWorkflowRunner._as_template_value(x) for x in v])
        if isinstance(v, dict):
            if "text" in v:
                return str(v["text"])
            if "document" in v:
                return str(v["document"])
            if "content" in v:
                return str(v["content"])
            if "title" in v and "snippet" in v:
                return f"{v['title']}: {v['snippet']}"
            return str(v)
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

    @staticmethod
    def _dict_lookup(value: Any, dotted: str) -> Any:
        cur = value
        for key in str(dotted).split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    def _extract_from_batch(self, query_batch: DataProto, key_or_path: str) -> Any:
        key_or_path = str(key_or_path)
        if key_or_path in query_batch.non_tensor_batch:
            vec = query_batch.non_tensor_batch[key_or_path]
            return vec[0] if len(vec) > 0 else None

        parts = key_or_path.split(".")
        if len(parts) <= 1:
            return None
        root = parts[0]
        if root not in query_batch.non_tensor_batch:
            return None
        root_vec = query_batch.non_tensor_batch[root]
        if len(root_vec) == 0:
            return None
        return self._dict_lookup(root_vec[0], ".".join(parts[1:]))

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, dict):
            for key in ("ground_truth", "answer", "target", "golden_answers"):
                if key in value:
                    return GraphWorkflowRunner._to_str_list(value[key])
            return [str(value)]
        if isinstance(value, np.ndarray):
            return [str(x).strip() for x in value.tolist() if str(x).strip()]
        if isinstance(value, list | tuple):
            return [str(x).strip() for x in value if str(x).strip()]
        v = str(value).strip()
        return [v] if v else []

    @staticmethod
    def _extract_question_from_messages(messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = []
                for x in content:
                    if isinstance(x, dict):
                        if x.get("type") == "text":
                            texts.append(str(x.get("text", "")))
                        elif "text" in x:
                            texts.append(str(x.get("text", "")))
                    else:
                        texts.append(str(x))
                merged = "".join(texts).strip()
                if merged:
                    return merged
        return ""

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
        if parser_type in {"tag", "tag_between"}:
            tag = str(parser.get("tag", "")).strip()
            open_tag = str(parser.get("open_tag", f"<{tag}>" if tag else "")).strip()
            close_tag = str(parser.get("close_tag", f"</{tag}>" if tag else "")).strip()
            valid_reward = float(parser.get("valid_reward", 0.0))
            invalid_reward = float(parser.get("invalid_reward", -1.0))
            if not open_tag or not close_tag:
                return raw_text.strip(), invalid_reward
            start_idx = raw_text.find(open_tag)
            if start_idx < 0:
                return raw_text.strip(), invalid_reward
            start_idx += len(open_tag)
            end_idx = raw_text.find(close_tag, start_idx)
            if end_idx < 0:
                return raw_text.strip(), invalid_reward
            value = raw_text[start_idx:end_idx].strip()
            if not value:
                return raw_text.strip(), invalid_reward
            return value, valid_reward
        value = raw_text.strip()
        return value, 1.0 if value else 0.0

    def _extract_question(self, query_batch: DataProto) -> str:
        for key in self.question_candidates:
            value = self._extract_from_batch(query_batch, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        prompt = self._extract_from_batch(query_batch, "prompt")
        parsed = self._extract_question_from_messages(prompt)
        if parsed:
            return parsed
        raw_prompt = self._extract_from_batch(query_batch, "raw_prompt")
        parsed = self._extract_question_from_messages(raw_prompt)
        if parsed:
            return parsed
        return ""

    def _extract_gt_list(self, query_batch: DataProto) -> list[str]:
        for key in self.gt_candidates:
            value = self._extract_from_batch(query_batch, key)
            gts = self._to_str_list(value)
            if gts:
                return gts
        return []

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
            action_text_vec = thin.non_tensor_batch.get("action_text", np.array([], dtype=object))
            raw_text = str(action_text_vec[0]) if len(action_text_vec) > 0 else ""
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
            top_k = int(node_cfg.get("top_k", 3))
            if hasattr(tool, "query"):
                max_attempts = int(node_cfg.get("max_attempts", 5))
                # Prefer legacy named args used by RetrievalTool(question, N, max_attempts),
                # then fallback to positional for custom tool implementations.
                try:
                    output = await asyncio.to_thread(
                        tool.query,
                        question=input_text,
                        N=top_k,
                        max_attempts=max_attempts,
                    )
                except TypeError:
                    output = await asyncio.to_thread(tool.query, input_text, top_k, max_attempts)
            elif hasattr(tool, "retrieve"):
                output = await asyncio.to_thread(tool.retrieve, input_text, top_k)
            elif callable(tool):
                output = await asyncio.to_thread(tool, input_text)
            else:
                raise TypeError(f"tool {tool_name} is not callable and has no query()/retrieve()")
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
        outcome_type = str(self.outcome_cfg.get("type", "em")).lower()
        source = str(self.outcome_cfg.get("source", ""))
        weight = float(self.outcome_cfg.get("weight", 1.0))
        pred = str(self._lookup_path(context, source, default=""))
        gt = context.get("ground_truth", [])
        if len(gt) <= 0:
            return 0.0
        if outcome_type == "em":
            return weight * float(em_check(pred, gt))
        if outcome_type == "f1":
            return weight * float(max(self._f1_score(pred, str(ans)) for ans in gt))
        return 0.0

    @staticmethod
    def _normalize_text(text: str) -> str:
        s = str(text or "").lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @classmethod
    def _f1_score(cls, pred: str, gt: str) -> float:
        pred_tokens = cls._normalize_text(pred).split()
        gt_tokens = cls._normalize_text(gt).split()
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gt_tokens)
        overlap = sum(common.values())
        if overlap <= 0:
            return 0.0
        precision = overlap / float(len(pred_tokens))
        recall = overlap / float(len(gt_tokens))
        return 2.0 * precision * recall / (precision + recall)

    async def _run_one_query(
        self,
        query_batch: DataProto,
        query_sem: asyncio.Semaphore,
        query_local_idx: int,
        debug_query_idx: int | None,
        debug_batch_idx: int,
    ) -> dict[str, Any]:
        async with query_sem:
            debug_lines: list[str] = []
            debug_on = self.debug_enabled and debug_query_idx is not None and query_local_idx == debug_query_idx

            context = {
                "question": self._extract_question(query_batch),
                "ground_truth": self._extract_gt_list(query_batch),
                "nodes": {},
                "step": 0,
            }

            if debug_on:
                query_id = self._extract_from_batch(query_batch, "query_id")
                debug_lines.append(
                    f"[star-debug] batch={debug_batch_idx} query_idx={query_local_idx} query_id={query_id}"
                )
                debug_lines.append(f"[star-debug] question={self._clip_debug_text(context['question'])}")
                debug_lines.append(
                    f"[star-debug] ground_truth={self._summarize_debug_value(context['ground_truth'])}"
                )

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
                    if debug_on:
                        if result["node_type"] == "llm":
                            debug_lines.append(
                                f"[star-debug] step={step} node={node_id} "
                                f"out={result['output_key']}:{self._summarize_debug_value(result['output_value'])} "
                                f"format={float(result['format_reward']):.2f}"
                            )
                        else:
                            debug_lines.append(
                                f"[star-debug] step={step} node={node_id} "
                                f"out={result['output_key']}:{self._summarize_debug_value(result['output_value'])}"
                            )
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
            debug_dump = None
            if debug_on:
                outcome_source = str(self.outcome_cfg.get("source", "") or "")
                if outcome_source:
                    pred = self._lookup_path(context, outcome_source, default="")
                    debug_lines.append(
                        f"[star-debug] final={outcome_source}:{self._summarize_debug_value(pred)}"
                    )
                debug_lines.append(
                    f"[star-debug] outcome_reward={float(outcome_reward):.4f} llm_nodes={len(llm_exec_records)}"
                )
                debug_dump = "\n".join(
                    [
                        "[star-debug] ===== trace begin =====",
                        *debug_lines,
                        "[star-debug] ===== trace end =====",
                    ]
                )

            reward_parts = []
            node_format = {}
            for rec in llm_exec_records:
                node_id = rec["node_id"]
                reward_cfg = dict(self.nodes[node_id].get("reward", {}))
                format_weight = float(reward_cfg.get("format_weight", 0.0))
                share_outcome = bool(reward_cfg.get("share_outcome", True))
                total = format_weight * float(rec["format_reward"]) + (outcome_reward if share_outcome else 0.0)
                thin_len = len(rec["thin"])
                if thin_len > 0:
                    reward_parts.append(
                        self.trainer._build_commit_rewards_from_thin(
                            rec["thin"], np.full((thin_len,), total, dtype=np.float32)
                        )
                    )
                node_format[node_id] = float(rec["format_reward"])

            return {
                "reward_parts": reward_parts,
                "outcome_reward": float(outcome_reward),
                "node_format": node_format,
                "llm_node_count": float(len(llm_exec_records)),
                "debug_dump": debug_dump,
            }

    async def run_batch(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        del epoch
        self._debug_batch_counter += 1
        debug_this_batch = self.debug_enabled and (
            self._debug_batch_counter % max(1, self.debug_every_n_batches) == 0
        )
        debug_query_idx = None
        if debug_this_batch and len(batch) > 0:
            debug_query_idx = int(self.debug_sample_index) % len(batch)

        query_sem = asyncio.Semaphore(max(1, self.max_inflight_queries))
        tasks = [
            self._run_one_query(
                batch.select_idxs([i]),
                query_sem,
                query_local_idx=i,
                debug_query_idx=debug_query_idx,
                debug_batch_idx=self._debug_batch_counter,
            )
            for i in range(len(batch))
        ]
        query_results = await asyncio.gather(*tasks)

        if debug_this_batch:
            for item in query_results:
                dump = item.get("debug_dump", None)
                if dump:
                    print(dump, flush=True)
                    break

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
