from __future__ import annotations

import ast
import asyncio
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from calosum.shared.utils.tools import ToolSchema

_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "help",
    "input",
    "open",
}
_BLOCKED_MODULE_ROOTS = {
    "httpx",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}
_SAFE_WRAPPER = """
from datetime import date, datetime, timedelta
import functools
import itertools
import json
import math
import random
import re
import statistics
import sys

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
namespace = {
    "__builtins__": SAFE_BUILTINS,
    "date": date,
    "datetime": datetime,
    "functools": functools,
    "itertools": itertools,
    "json": json,
    "math": math,
    "random": random,
    "re": re,
    "statistics": statistics,
    "timedelta": timedelta,
}
source = open(sys.argv[1], "r", encoding="utf-8").read()
exec(compile(source, "<code_execution>", "exec"), namespace, namespace)
""".strip()


@dataclass(slots=True)
class CodeExecutionTool:
    default_timeout_seconds: float = 2.0
    max_output_chars: int = 4000
    schema: ToolSchema = field(
        default_factory=lambda: ToolSchema(
            name="code_execution",
            description="Execute constrained Python code in a temporary isolated subprocess",
            parameters={"code": "string"},
            required_permissions=["process_exec"],
            needs_approval=True,
        )
    )

    async def execute(self, payload: dict[str, object], session_id: str | None = None) -> str:
        code = str(payload.get("code", "") or "")
        if not code.strip():
            return "Code execution rejected: empty code payload."

        policy_violations = _validate_code_policy(code)
        if policy_violations:
            return "Code execution rejected: " + "; ".join(policy_violations)

        timeout_seconds = _bounded_timeout(
            payload.get("timeout_seconds"),
            default=self.default_timeout_seconds,
        )
        return await _run_in_subprocess(
            code,
            timeout_seconds=timeout_seconds,
            max_output_chars=self.max_output_chars,
            session_id=session_id,
        )


def _validate_code_policy(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BLOCKED_MODULE_ROOTS or True:
                    violations.append(f"imports are not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            violations.append(f"imports are not allowed: {target}")
        elif isinstance(node, ast.Call):
            target = _call_name(node.func)
            if target in _BLOCKED_CALLS:
                violations.append(f"blocked call: {target}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in _BLOCKED_MODULE_ROOTS:
                violations.append(f"blocked module access: {node.value.id}.{node.attr}")

    return sorted(set(violations))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
    return None


def _bounded_timeout(value: object, *, default: float) -> float:
    try:
        timeout = float(value) if value is not None else default
    except (TypeError, ValueError):
        timeout = default
    return max(0.2, min(10.0, timeout))


async def _run_in_subprocess(
    code: str,
    *,
    timeout_seconds: float,
    max_output_chars: int,
    session_id: str | None = None,
) -> str:
    sandbox_base = Path(tempfile.gettempdir()) / "calosum_sandbox"
    if session_id:
        temp_path = sandbox_base / session_id / "python_exec"
    else:
        # Fallback para comportamento efêmero se não houver sessão
        temp_path = Path(tempfile.mkdtemp(prefix="calosum_tool_code_exec_"))
    
    temp_path.mkdir(parents=True, exist_ok=True)
    try:
        source_path = temp_path / "snippet.py"
        source_path.write_text(code, encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _SAFE_WRAPPER,
            str(source_path),
            cwd=str(temp_path),
            env={"PYTHONIOENCODING": "utf-8"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return f"Code execution timed out after {timeout_seconds:.1f}s."

        rendered_stdout = stdout.decode("utf-8", errors="replace").strip()
        rendered_stderr = stderr.decode("utf-8", errors="replace").strip()
        payload = _truncate_text(rendered_stdout or rendered_stderr or "(no output)", max_output_chars)

        if process.returncode == 0:
            return payload

        error_suffix = f" stderr={payload}" if payload else ""
        return f"Code exited with status {process.returncode}.{error_suffix}"
    finally:
        # Só remove se for efêmero (sem session_id)
        if not session_id and temp_path.exists():
            try:
                import shutil
                shutil.rmtree(temp_path)
            except Exception:
                pass


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n...[truncated]"
