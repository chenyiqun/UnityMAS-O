import json
from abc import ABC, abstractmethod
from pathlib import Path


class RetrieverToolInterface(ABC):
    """Tool interface for retrieval backend."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[str]:
        raise NotImplementedError


class HttpRetrieverTool(RetrieverToolInterface):
    """Placeholder retriever for external service integration."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def retrieve(self, query: str, top_k: int) -> list[str]:
        raise NotImplementedError(
            f"HTTP retriever interface is ready but not implemented. endpoint={self.endpoint}. "
            "Please implement POST /retrieve with payload {'query': str, 'top_k': int}."
        )


class SimpleKeywordRetrieverTool(RetrieverToolInterface):
    """Minimal local retriever for examples/tests.

    It scores docs by token-overlap count with the query.
    """

    def __init__(self, documents: list[str]):
        self.documents = documents

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.strip().lower() for t in text.split() if t.strip()}

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return self.documents[:top_k]

        scored = []
        for idx, doc in enumerate(self.documents):
            d_tokens = self._tokenize(doc)
            score = len(q_tokens.intersection(d_tokens))
            scored.append((score, -idx, doc))
        scored.sort(reverse=True)
        return [doc for _, _, doc in scored[:top_k]]


def _load_docs_from_path(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix in {".txt"}:
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    if p.suffix in {".jsonl"}:
        docs = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                docs.append(str(obj.get("text", obj.get("document", ""))).strip())
            else:
                docs.append(str(obj).strip())
        return [d for d in docs if d]

    if p.suffix in {".json"}:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            docs = []
            for x in obj:
                if isinstance(x, dict):
                    docs.append(str(x.get("text", x.get("document", ""))).strip())
                else:
                    docs.append(str(x).strip())
            return [d for d in docs if d]

    return []


def build_retriever_tool(retriever_cfg) -> RetrieverToolInterface:
    retriever_type = str(retriever_cfg.get("type", "simple_keyword"))

    if retriever_type == "http":
        endpoint = str(retriever_cfg.get("endpoint", "http://127.0.0.1:8000/retrieve"))
        return HttpRetrieverTool(endpoint=endpoint)

    docs = []
    for item in retriever_cfg.get("documents", []):
        docs.append(str(item))
    corpus_path = retriever_cfg.get("corpus_path", None)
    if corpus_path is not None:
        docs.extend(_load_docs_from_path(str(corpus_path)))
    docs = [x for x in docs if x]
    return SimpleKeywordRetrieverTool(documents=docs)
