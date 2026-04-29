from __future__ import annotations

import ast
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from typing import Any


class CodeVerifierTool:
    """Run generated Python code against code-task tests.

    This verifier keeps the current STAR workflow schema:
    input payload: {"problem": str, "code": str, "tests": Any, ...}
    output: pass_rate/all_passed/passed/total/error/failed_test_index.

    It also accepts MARTI/LiveCodeBench-style tests:
    {"inputs": [...], "outputs": [...], "fn_name": "..."}.
    If fn_name is present, the generated code is evaluated as a callable
    solution. Otherwise it is executed as a standard-input program.
    """

    _CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
    _PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_error_chars: int = 2000,
        python_executable: str | None = None,
        allow_custom_checker: bool = False,
        default_checker_type: str = "auto",
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.max_error_chars = int(max_error_chars)
        self.python_executable = python_executable or sys.executable
        self.allow_custom_checker = bool(allow_custom_checker)
        self.default_checker_type = str(default_checker_type or "auto")

    @classmethod
    def extract_code(cls, value: Any, min_fenced_length: int = 1, strict_syntax: bool = True) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""

        tag_match = cls._CODE_TAG_RE.search(raw)
        if tag_match:
            return tag_match.group(1).strip()

        valid_blocks: list[str] = []
        for block in cls._PY_FENCE_RE.findall(raw):
            code = str(block or "").strip()
            if len(code) < min_fenced_length:
                continue
            if strict_syntax:
                try:
                    ast.parse(code, mode="exec")
                except (SyntaxError, IndentationError):
                    continue
            valid_blocks.append(code)
        if valid_blocks:
            return valid_blocks[-1]

        return raw

    @staticmethod
    def _loads_maybe(value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return value

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        value = CodeVerifierTool._loads_maybe(value)
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def normalize_tests(cls, tests: Any) -> list[dict[str, Any]]:
        """Backward-compatible list-of-cases view."""

        return cls.normalize_test_spec(tests)["cases"]

    @classmethod
    def normalize_test_spec(cls, tests: Any) -> dict[str, Any]:
        tests = cls._loads_maybe(tests)
        if isinstance(tests, dict) and "tests" in tests:
            tests = cls._loads_maybe(tests["tests"])

        # MARTI answer schema: use public tests by default for training-time rewards.
        if isinstance(tests, dict) and ("public_tests" in tests or "private_tests" in tests):
            tests = cls._loads_maybe(tests.get("public_tests") or tests.get("private_tests") or {})

        common: dict[str, Any] = {}
        cases: list[dict[str, Any]] = []
        if isinstance(tests, dict):
            common = {
                "fn_name": tests.get("fn_name") or tests.get("function_name") or "",
                "checker_type": tests.get("checker_type") or tests.get("special_judge") or "",
                "checker_code": tests.get("checker_code") or tests.get("checker") or "",
            }
            if "inputs" in tests and "outputs" in tests:
                inputs = cls._as_list(tests.get("inputs"))
                outputs = cls._as_list(tests.get("outputs"))
                cases = [{"input": inp, "output": out, **common} for inp, out in zip(inputs, outputs)]
            elif "input" in tests and "output" in tests:
                cases = [{"input": tests.get("input", ""), "output": tests.get("output", ""), **common}]

        elif isinstance(tests, list):
            inferred_fn_name = ""
            inferred_checker_type = ""
            inferred_checker_code = ""
            for item in tests:
                item = cls._loads_maybe(item)
                if not isinstance(item, dict):
                    continue
                inferred_fn_name = inferred_fn_name or item.get("fn_name") or item.get("function_name") or ""
                inferred_checker_type = (
                    inferred_checker_type or item.get("checker_type") or item.get("special_judge") or ""
                )
                inferred_checker_code = inferred_checker_code or item.get("checker_code") or item.get("checker") or ""
                if "input" in item or "output" in item:
                    inp = item.get("input", item.get("stdin", ""))
                    out = item.get("output", item.get("stdout", ""))
                    cases.append(
                        {
                            "input": inp,
                            "output": out,
                            "fn_name": item.get("fn_name") or item.get("function_name") or "",
                            "checker_type": item.get("checker_type") or item.get("special_judge") or "",
                            "checker_code": item.get("checker_code") or item.get("checker") or "",
                        }
                    )
                elif "inputs" in item and "outputs" in item:
                    item_common = {
                        "fn_name": item.get("fn_name") or item.get("function_name") or "",
                        "checker_type": item.get("checker_type") or item.get("special_judge") or "",
                        "checker_code": item.get("checker_code") or item.get("checker") or "",
                    }
                    for inp, out in zip(cls._as_list(item["inputs"]), cls._as_list(item["outputs"])):
                        cases.append({"input": inp, "output": out, **item_common})
            common = {
                "fn_name": inferred_fn_name,
                "checker_type": inferred_checker_type,
                "checker_code": inferred_checker_code,
            }

        for case in cases:
            for key in ("fn_name", "checker_type", "checker_code"):
                if not case.get(key) and common.get(key):
                    case[key] = common[key]
            if isinstance(case.get("input"), list):
                case["input"] = "\n".join(str(x) for x in case["input"])
            if isinstance(case.get("output"), list) and not case.get("fn_name"):
                case["output"] = "\n".join(str(x) for x in case["output"])

        return {"cases": cases, **common}

    @staticmethod
    def _guard_prelude() -> str:
        return textwrap.dedent(
            """
            import builtins as __verl_builtins
            __verl_builtins.exit = None
            __verl_builtins.quit = None
            try:
                import os as __verl_os
                __verl_os.environ["OMP_NUM_THREADS"] = "1"
                for __verl_name in (
                    "kill", "system", "putenv", "remove", "removedirs", "rmdir",
                    "fchdir", "setuid", "fork", "forkpty", "killpg", "rename",
                    "renames", "truncate", "replace", "unlink", "fchmod",
                    "fchown", "chmod", "chown", "chroot", "lchflags",
                    "lchmod", "lchown", "chdir",
                ):
                    if hasattr(__verl_os, __verl_name):
                        setattr(__verl_os, __verl_name, None)
            except Exception:
                pass
            try:
                import shutil as __verl_shutil
                for __verl_name in ("rmtree", "move", "chown"):
                    if hasattr(__verl_shutil, __verl_name):
                        setattr(__verl_shutil, __verl_name, None)
            except Exception:
                pass
            try:
                import subprocess as __verl_subprocess
                __verl_subprocess.Popen = None
            except Exception:
                pass
            """
        ).strip()

    @staticmethod
    def _preexec_memory_limit() -> Any:
        if platform.system() == "Darwin":
            return None

        def _limit() -> None:
            try:
                import resource

                max_bytes = 512 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
                resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
            except Exception:
                pass

        return _limit

    @staticmethod
    def _json_safe_expr() -> str:
        return textwrap.dedent(
            """
            def __verl_json_safe(value):
                if isinstance(value, tuple):
                    return [__verl_json_safe(x) for x in value]
                if isinstance(value, list):
                    return [__verl_json_safe(x) for x in value]
                if isinstance(value, dict):
                    return {str(k): __verl_json_safe(v) for k, v in value.items()}
                return value
            """
        ).strip()

    def _write_solution(self, tmpdir: str, code: str) -> str:
        solution_path = os.path.join(tmpdir, "solution.py")
        source = self._guard_prelude() + "\n\n" + code + "\n"
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(source)
        return solution_path

    @staticmethod
    def _strip_main_guard(code: str) -> str:
        try:
            tree = ast.parse(str(code or ""))
            body = []
            changed = False
            for node in tree.body:
                if isinstance(node, ast.If):
                    try:
                        condition = ast.unparse(node.test).replace('"', "'").strip()
                    except Exception:
                        condition = ""
                    if condition == "__name__ == '__main__'":
                        changed = True
                        continue
                body.append(node)
            if not changed:
                return code
            tree.body = body
            return ast.unparse(tree)
        except Exception:
            return code

    def _write_call_runner(self, tmpdir: str, code: str, fn_name: str) -> str:
        runner_path = os.path.join(tmpdir, "call_runner.py")
        code = self._strip_main_guard(code)
        runner = (
            self._guard_prelude()
            + "\n\n"
            + code
            + "\n\n"
            + self._json_safe_expr()
            + "\n"
            + textwrap.dedent(
                f"""
                import json as __verl_json
                import sys as __verl_sys
                __verl_payload = __verl_sys.stdin.read()
                __verl_lines = [line for line in __verl_payload.splitlines() if line.strip()]
                __verl_args = [__verl_json.loads(line) for line in __verl_lines]
                if not __verl_args and __verl_payload.strip():
                    __verl_args = [__verl_json.loads(__verl_payload)]
                __verl_target = globals().get({fn_name!r})
                if __verl_target is None and "Solution" in globals():
                    __verl_obj = globals()["Solution"]()
                    __verl_target = getattr(__verl_obj, {fn_name!r})
                if __verl_target is None:
                    raise AttributeError("function not found: " + {fn_name!r})
                __verl_output = __verl_target(*__verl_args)
                __verl_sys.stdout.write(__verl_json.dumps(__verl_json_safe(__verl_output), ensure_ascii=False))
                """
            ).strip()
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner)
        return runner_path

    def _run_python(self, script_path: str, stdin: str, cwd: str) -> tuple[int, str, str, str]:
        try:
            proc = subprocess.run(
                [self.python_executable, script_path],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=cwd,
                preexec_fn=self._preexec_memory_limit(),
            )
            return proc.returncode, proc.stdout, proc.stderr, ""
        except subprocess.TimeoutExpired:
            return -1, "", "", f"Time limit exceeded after {self.timeout_seconds:.2f}s."
        except Exception as exc:
            return -1, "", "", f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _parse_json_maybe(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except Exception:
            return value

    @classmethod
    def _objects_equal(cls, actual: Any, expected: Any) -> bool:
        actual = cls._parse_json_maybe(actual)
        expected = cls._parse_json_maybe(expected)
        if isinstance(actual, tuple):
            actual = list(actual)
        if isinstance(expected, tuple):
            expected = list(expected)
        if actual == expected:
            return True
        if isinstance(expected, list) and len(expected) == 1 and actual == expected[0]:
            return True
        if isinstance(actual, list) and isinstance(expected, list):
            return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(
                expected, sort_keys=True, ensure_ascii=False
            )
        return False

    @staticmethod
    def _float_tokens_close(actual_tokens: list[str], expected_tokens: list[str]) -> bool:
        if len(actual_tokens) != len(expected_tokens):
            return False
        try:
            actual_f = [float(x) for x in actual_tokens]
            expected_f = [float(x) for x in expected_tokens]
        except Exception:
            return False
        return all(abs(a - e) <= 1e-6 * max(1.0, abs(e)) for a, e in zip(actual_f, expected_f))

    @staticmethod
    def _yes_no_case_insensitive_match(actual_lines: list[str], expected_lines: list[str]) -> bool:
        if len(actual_lines) != len(expected_lines):
            return False
        yn = {"yes", "no"}
        for a, e in zip(actual_lines, expected_lines):
            if e.lower() not in yn:
                return False
            if a.lower() != e.lower():
                return False
        return True

    @classmethod
    def _standard_output_match(cls, actual: str, expected: str) -> bool:
        actual_s = str(actual or "").strip()
        expected_s = str(expected or "").strip()
        if actual_s == expected_s:
            return True

        actual_lines = [line.strip() for line in actual_s.splitlines() if line.strip()]
        expected_lines = [line.strip() for line in expected_s.splitlines() if line.strip()]
        if actual_lines == expected_lines:
            return True
        if cls._yes_no_case_insensitive_match(actual_lines, expected_lines):
            return True

        actual_tokens = actual_s.split()
        expected_tokens = expected_s.split()
        if actual_tokens == expected_tokens:
            return True
        if cls._float_tokens_close(actual_tokens, expected_tokens):
            return True
        return False

    @staticmethod
    def _infer_checker_type(problem: str, case: dict[str, Any], default_checker_type: str) -> str:
        explicit = str(case.get("checker_type") or "").strip()
        if explicit:
            return explicit
        if default_checker_type and default_checker_type != "auto":
            return default_checker_type
        p = str(problem or "").lower()
        if "rearrange the characters" in p and "not equal to" in p and "impossible" in p:
            return "cf_rearrange_string"
        return "standard"

    @staticmethod
    def _check_cf_rearrange_string(input_data: str, actual_output: str) -> tuple[bool, str]:
        tokens = str(input_data or "").split()
        if not tokens:
            return False, "empty input"
        try:
            t = int(tokens[0])
        except Exception:
            return False, "cannot parse number of test cases"
        strings = tokens[1 : 1 + t]
        if len(strings) != t:
            return False, f"expected {t} strings, got {len(strings)}"

        out_lines = [line.strip() for line in str(actual_output or "").splitlines() if line.strip()]
        ptr = 0
        for case_idx, s in enumerate(strings):
            impossible = len(set(s)) <= 1
            if ptr >= len(out_lines):
                return False, f"missing verdict for case {case_idx}"
            verdict = out_lines[ptr].lower()
            ptr += 1
            if impossible:
                if verdict != "no":
                    return False, f"case {case_idx}: expected NO"
                continue
            if verdict != "yes":
                return False, f"case {case_idx}: expected YES"
            if ptr >= len(out_lines):
                return False, f"case {case_idx}: missing rearranged string"
            r = out_lines[ptr]
            ptr += 1
            if len(r) != len(s) or Counter(r) != Counter(s):
                return False, f"case {case_idx}: output is not a permutation"
            if r == s:
                return False, f"case {case_idx}: output equals original string"
        if ptr != len(out_lines):
            return False, "extra output lines"
        return True, ""

    def _run_custom_checker(
        self,
        checker_code: str,
        input_data: str,
        expected_output: str,
        actual_output: str,
        problem: str,
        metadata: Any,
    ) -> tuple[bool, str]:
        if not self.allow_custom_checker:
            return False, "custom checker present but allow_custom_checker=false"
        ns: dict[str, Any] = {}
        try:
            exec(str(checker_code), ns)
            check_fn = ns.get("check")
            if not callable(check_fn):
                return False, "checker_code must define check(input_data, expected_output, actual_output, problem, metadata)"
            ok = check_fn(input_data, expected_output, actual_output, problem, metadata)
            return bool(ok), "" if ok else "custom checker returned false"
        except Exception as exc:
            return False, f"custom checker error: {type(exc).__name__}: {exc}"

    def _outputs_match(
        self,
        *,
        actual: str,
        expected: Any,
        input_data: Any,
        problem: str,
        case: dict[str, Any],
        metadata: Any,
        call_based: bool,
    ) -> tuple[bool, str, str]:
        if call_based:
            ok = self._objects_equal(actual, expected)
            return ok, "call_based", "" if ok else "call-based output mismatch"

        checker_type = self._infer_checker_type(problem, case, self.default_checker_type)
        checker_code = str(case.get("checker_code") or "")
        if checker_code:
            ok, reason = self._run_custom_checker(
                checker_code,
                str(input_data or ""),
                str(expected or ""),
                str(actual or ""),
                problem,
                metadata,
            )
            return ok, "custom_checker", reason

        if checker_type == "cf_rearrange_string":
            ok, reason = self._check_cf_rearrange_string(str(input_data or ""), str(actual or ""))
            return ok, checker_type, reason
        if checker_type == "unordered_tokens":
            ok = Counter(str(actual or "").split()) == Counter(str(expected or "").split())
            return ok, checker_type, "" if ok else "unordered token multiset mismatch"

        ok = self._standard_output_match(str(actual or ""), str(expected or ""))
        return ok, checker_type, "" if ok else "standard output mismatch"

    def _truncate(self, value: Any) -> str:
        text = str(value or "")
        if len(text) <= self.max_error_chars:
            return text
        keep = max(0, self.max_error_chars // 2)
        return text[:keep] + "\n...(truncated)...\n" + text[-keep:]

    def __call__(self, payload: Any) -> dict[str, Any]:
        start = time.perf_counter()
        payload = payload if isinstance(payload, dict) else {}
        problem = str(payload.get("problem", "") or "")
        metadata = payload.get("metadata", payload.get("extra_info", {}))
        code = self.extract_code(payload.get("code", ""))
        tests = self.normalize_tests(payload.get("tests", []))
        if isinstance(metadata, str):
            metadata = self._loads_maybe(metadata)
        if isinstance(metadata, dict):
            metadata_fn_name = str(metadata.get("func_name") or metadata.get("function_name") or "").strip()
            if metadata_fn_name:
                for test in tests:
                    if not test.get("fn_name"):
                        test["fn_name"] = metadata_fn_name

        result: dict[str, Any] = {
            "pass_rate": 0.0,
            "all_passed": 0,
            "passed": 0,
            "total": len(tests),
            "error": "",
            "error_code": 0,
            "error_message": "",
            "failed_test_index": -1,
            "test_results": [],
            "checker_type": "",
            "duration_s": 0.0,
        }
        if not code.strip():
            result["error"] = "No code found."
            result["error_code"] = -1
            result["error_message"] = "No code found"
            result["duration_s"] = float(time.perf_counter() - start)
            return result
        if not tests:
            result["error"] = "No tests found."
            result["error_code"] = -1
            result["error_message"] = "No tests found"
            result["duration_s"] = float(time.perf_counter() - start)
            return result

        hard_deadline = start + self.timeout_seconds * max(1, len(tests)) + 5.0
        passed = 0
        first_error = ""
        first_error_code = 0
        first_error_message = ""
        first_failed_idx = -1
        checker_types: list[str] = []

        with tempfile.TemporaryDirectory(prefix="verl_code_verify_") as tmpdir:
            solution_path = self._write_solution(tmpdir, code)
            call_runner_by_fn: dict[str, str] = {}

            for idx, test in enumerate(tests):
                failure_kind = ""
                if time.perf_counter() > hard_deadline:
                    ok = False
                    stdout = ""
                    stderr = ""
                    run_error = "Global verifier timeout."
                    returncode = -1
                    failure_kind = "timeout"
                else:
                    fn_name = str(test.get("fn_name") or "").strip()
                    stdin = str(test.get("input", ""))
                    if fn_name:
                        if fn_name not in call_runner_by_fn:
                            call_runner_by_fn[fn_name] = self._write_call_runner(tmpdir, code, fn_name)
                        returncode, stdout, stderr, run_error = self._run_python(
                            call_runner_by_fn[fn_name],
                            stdin,
                            tmpdir,
                        )
                    else:
                        returncode, stdout, stderr, run_error = self._run_python(solution_path, stdin, tmpdir)

                    if run_error:
                        ok = False
                        failure_kind = "timeout" if "time" in run_error.lower() else "runtime"
                    elif returncode != 0:
                        ok = False
                        failure_kind = "runtime"
                    else:
                        ok, checker_type, reason = self._outputs_match(
                            actual=stdout,
                            expected=test.get("output", ""),
                            input_data=test.get("input", ""),
                            problem=problem,
                            case=test,
                            metadata=metadata,
                            call_based=bool(fn_name),
                        )
                        checker_types.append(checker_type)
                        if not ok:
                            run_error = reason
                            failure_kind = "wrong_answer"

                result["test_results"].append(bool(ok))
                if ok:
                    passed += 1
                    continue

                if first_failed_idx < 0:
                    first_failed_idx = idx
                    if failure_kind in {"timeout", "runtime"} and run_error:
                        first_error_code = -3 if failure_kind == "timeout" else -4
                        first_error_message = "Time Limit Exceeded" if failure_kind == "timeout" else "Runtime Error"
                        first_error = run_error
                    elif returncode != 0:
                        first_error_code = -4
                        first_error_message = "Runtime Error"
                        first_error = stderr or f"Non-zero exit code: {returncode}"
                    else:
                        first_error_code = -2
                        first_error_message = "Wrong Answer"
                        first_error = (
                            f"{run_error or 'Wrong Answer'}\n"
                            f"Input:\n{test.get('input', '')}\n\n"
                            f"Expected:\n{test.get('output', '')}\n\n"
                            f"Actual:\n{stdout}"
                        )

        total = max(1, len(tests))
        result["passed"] = int(passed)
        result["pass_rate"] = float(passed / total)
        result["all_passed"] = int(passed == len(tests))
        result["failed_test_index"] = int(first_failed_idx)
        result["error"] = self._truncate(first_error)
        result["error_code"] = int(first_error_code)
        result["error_message"] = first_error_message
        result["checker_type"] = ",".join(sorted(set(checker_types))) if checker_types else ""
        result["duration_s"] = float(time.perf_counter() - start)
        return result

    async def acall(self, payload: Any) -> dict[str, Any]:
        """Async tool entrypoint used by STAR workflow runners."""

        return await asyncio.to_thread(self.__call__, payload)
