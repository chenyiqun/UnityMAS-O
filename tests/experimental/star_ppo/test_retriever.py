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

from urllib import error

import pytest

from verl.experimental.star_ppo.tools.retriever import HttpRetrieverTool


def test_random_endpoint_routing_tries_configured_replicas_before_fallback_paths(monkeypatch):
    monkeypatch.setenv("STAR_RETRIEVER_RANDOM_ENDPOINT", "true")
    configured_urls = [
        "http://retriever:8000/retrieve",
        "http://retriever:8001/retrieve",
    ]
    tool = HttpRetrieverTool(configured_urls, timeout_seconds=1)
    calls = []

    def fake_post(url, payload, timeout_seconds):
        calls.append((url, payload, timeout_seconds))
        return [{"top_k_docs": ["Paris"]}]

    monkeypatch.setattr(tool, "_post_json", fake_post)

    assert tool.retrieve("France capital", top_k=1) == ["Paris"]
    assert calls[0][0] in configured_urls


def test_retriever_tries_compatible_payload_on_same_endpoint(monkeypatch):
    tool = HttpRetrieverTool(["http://retriever:8000/retrieve"], timeout_seconds=1)
    calls = []

    def fake_post(url, payload, timeout_seconds):
        calls.append((url, payload, timeout_seconds))
        if "questions" in payload:
            raise ValueError("unsupported payload")
        return {"result": [[{"document": "Paris"}]]}

    monkeypatch.setattr(tool, "_post_json", fake_post)

    assert tool.retrieve("France capital", top_k=1) == ["Paris"]
    assert [sorted(payload) for _, payload, _ in calls] == [
        ["N", "questions"],
        ["queries", "return_scores", "topk"],
    ]


def test_retriever_max_attempts_bounds_failed_endpoint_calls(monkeypatch):
    tool = HttpRetrieverTool(
        [
            "http://retriever:8000/retrieve",
            "http://retriever:8001/retrieve",
            "http://retriever:8002/retrieve",
        ],
        timeout_seconds=1,
    )
    calls = []

    def fake_post(url, payload, timeout_seconds):
        calls.append((url, payload, timeout_seconds))
        raise error.URLError("offline")

    monkeypatch.setattr(tool, "_post_json", fake_post)

    with pytest.raises(RuntimeError, match="2 endpoint attempts"):
        tool.query("France capital", N=1, max_attempts=2)

    assert len(calls) == 2
