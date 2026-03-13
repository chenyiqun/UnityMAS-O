import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse


class RetrieverToolInterface(ABC):
    """Tool interface for retrieval backend."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[str]:
        raise NotImplementedError


class HttpRetrieverTool(RetrieverToolInterface):
    """HTTP retrieval adapter for query(question, N) style APIs."""

    def __init__(self, api_urls: list[str], timeout_seconds: float = 5.0):
        if not api_urls:
            raise ValueError("api_urls list cannot be empty")
        expanded_urls: list[str] = []
        for raw_url in api_urls:
            expanded_urls.extend(self._expand_candidate_urls(str(raw_url)))
        # keep order, remove duplicates
        self.api_urls = list(dict.fromkeys(expanded_urls))
        if len(self.api_urls) == 0:
            raise ValueError("api_urls must contain at least one valid URL")
        self.timeout_seconds = float(timeout_seconds)
        # Sticky endpoint routing:
        # - cache the last successful endpoint
        # - permanently skip endpoints that return 404
        self._preferred_url: str | None = None
        self._bad_urls_404: set[str] = set()
        self._state_lock = threading.Lock()

    @staticmethod
    def _expand_candidate_urls(raw_url: str) -> list[str]:
        """Expand one config item to candidate endpoints.

        Supports:
        - full endpoint URL (e.g., http://host:8000/retrieve)
        - host only/base URL (e.g., http://host:8000), then auto-try common paths.
        """
        url = str(raw_url).strip()
        if not url:
            return []

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return [url]

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").rstrip("/")

        if path and path != "":
            # If user already provided a concrete endpoint, keep it first.
            candidates = [url]
            # Compatibility fallback for services that don't expose /retrieve.
            if path.lower() == "/retrieve":
                candidates.extend([f"{base}/query", f"{base}/search"])
            return candidates

        # Base URL only: try common endpoint names.
        return [f"{base}/retrieve", f"{base}/query", f"{base}/search", base]

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> Any:
        req = request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    @staticmethod
    def _extract_top_k_docs(resp_obj: Any) -> list[dict[str, Any]]:
        # Format A (legacy expected):
        # [{"top_k_docs": [...]}, ...]
        if isinstance(resp_obj, list) and len(resp_obj) > 0 and isinstance(resp_obj[0], dict):
            docs = resp_obj[0].get("top_k_docs", None)
            if isinstance(docs, list):
                out = []
                for item in docs:
                    if isinstance(item, dict):
                        out.append(item)
                    else:
                        out.append({"text": str(item)})
                return out

        # Format B (some retriever servers):
        # {"result": [[{"document": "..."} ...], ...]}
        if isinstance(resp_obj, dict):
            result = resp_obj.get("result", None)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                out = []
                for item in result[0]:
                    if isinstance(item, dict):
                        text = item.get("text", item.get("document", item.get("content", None)))
                        if text is not None:
                            out.append({"text": str(text)})
                        else:
                            out.append({"text": json.dumps(item, ensure_ascii=False)})
                    else:
                        out.append({"text": str(item)})
                if len(out) > 0:
                    return out

            docs = resp_obj.get("top_k_docs", None)
            if isinstance(docs, list):
                out = []
                for item in docs:
                    if isinstance(item, dict):
                        out.append(item)
                    else:
                        out.append({"text": str(item)})
                return out

        raise ValueError("Unexpected API response format: expecting list[{'top_k_docs': list}]")

    def query(self, question: str, N: int, max_attempts: int = 5) -> list[dict[str, Any]]:
        payload_candidates = [
            {"questions": [str(question)], "N": int(N)},
            {"queries": [str(question)], "topk": int(N), "return_scores": False},
        ]
        last_error = ""
        attempts = 0

        while attempts < max_attempts:
            with self._state_lock:
                preferred = self._preferred_url
                bad_404 = set(self._bad_urls_404)

            candidate_urls: list[str] = []
            if preferred and preferred not in bad_404:
                candidate_urls.append(preferred)
            for url in self.api_urls:
                if url in bad_404:
                    continue
                if url not in candidate_urls:
                    candidate_urls.append(url)

            # If all urls were marked bad (e.g. service updated), allow re-probing.
            if len(candidate_urls) == 0:
                candidate_urls = list(self.api_urls)

            for api_url in candidate_urls:
                for payload in payload_candidates:
                    try:
                        data = self._post_json(api_url, payload, timeout_seconds=self.timeout_seconds)
                        docs = self._extract_top_k_docs(data)
                        with self._state_lock:
                            self._preferred_url = api_url
                        return docs
                    except error.HTTPError as e:
                        # 404 means wrong endpoint path; blacklist it to avoid log spam.
                        if int(getattr(e, "code", 0)) == 404:
                            with self._state_lock:
                                self._bad_urls_404.add(api_url)
                        last_error = (
                            f"url={api_url}, payload_keys={list(payload.keys())}, "
                            f"err={type(e).__name__}: {e}"
                        )
                        continue
                    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
                        last_error = (
                            f"url={api_url}, payload_keys={list(payload.keys())}, "
                            f"err={type(e).__name__}: {e}"
                        )
                        continue
            attempts += 1

        suffix = f", last_error: {last_error}" if last_error else ""
        raise RuntimeError(f"Request failed after {max_attempts} attempts{suffix}")

    def retrieve(self, query: str, top_k: int) -> list[str]:
        docs = self.query(question=query, N=top_k)
        out: list[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                out.append(str(doc))
                continue
            text = doc.get("text", None)
            if text is None:
                text = doc.get("document", doc.get("content", json.dumps(doc, ensure_ascii=False)))
            out.append(str(text))
        return out


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

    if retriever_type in {"http", "query_api_pool", "retrieval_api_pool"}:
        api_urls = list(retriever_cfg.get("api_urls", []))
        endpoint = retriever_cfg.get("endpoint", None)
        if endpoint is not None and len(api_urls) == 0:
            api_urls = [str(endpoint)]
        timeout_seconds = float(retriever_cfg.get("timeout_seconds", 5.0))
        return HttpRetrieverTool(api_urls=api_urls, timeout_seconds=timeout_seconds)

    docs = []
    for item in retriever_cfg.get("documents", []):
        docs.append(str(item))
    corpus_path = retriever_cfg.get("corpus_path", None)
    if corpus_path is not None:
        docs.extend(_load_docs_from_path(str(corpus_path)))
    docs = [x for x in docs if x]
    return SimpleKeywordRetrieverTool(documents=docs)


# Alias to align with externally referenced class name.
RetrievalTool = HttpRetrieverTool
