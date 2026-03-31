from __future__ import annotations

from typing import Any


class SubQEvidenceContextBuilder:
    """Build dynamic sub-question/evidence context text from workflow node outputs."""

    @staticmethod
    def _to_clean_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text

    def __call__(self, value: Any) -> str:
        nodes = value if isinstance(value, dict) else {}
        decompose_node = nodes.get("decompose", {}) if isinstance(nodes.get("decompose", {}), dict) else {}
        sub_queries_raw = decompose_node.get("sub_queries", [])

        if isinstance(sub_queries_raw, (list, tuple)):
            sub_queries = [self._to_clean_text(x) for x in sub_queries_raw]
        elif isinstance(sub_queries_raw, str):
            cleaned = self._to_clean_text(sub_queries_raw)
            sub_queries = [cleaned] if cleaned else []
        else:
            sub_queries = []

        blocks: list[str] = []
        for idx, sub_q in enumerate(sub_queries, start=1):
            if not sub_q:
                continue
            evidence_node = nodes.get(f"evidence_{idx}", {})
            if isinstance(evidence_node, dict):
                sub_ev = self._to_clean_text(evidence_node.get("sub_evidence", ""))
            else:
                sub_ev = ""
            if not sub_ev:
                sub_ev = "Unknown"
            blocks.append(f"Sub-question {idx}:\n{sub_q}\nSub-evidence {idx}:\n{sub_ev}")

        if len(blocks) == 0:
            return "No valid sub-question/evidence is available."

        return "\n\n".join(blocks)
