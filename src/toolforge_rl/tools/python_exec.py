"""Small, deterministic Python calculator used by experiment trajectories.

This is an experiment boundary, not a general untrusted-code sandbox.  It uses a
separate isolated interpreter, an AST policy, resource limits, and a short timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PythonExecResult:
    ok: bool
    output: str
    error_type: str | None
    return_code: int | None


_WRAPPER = r'''
import ast, builtins, json, resource, sys

code = sys.stdin.read()
allowed_modules = {"math", "statistics", "fractions", "decimal", "itertools", "functools", "collections"}
banned_names = {"eval", "exec", "open", "compile", "input", "globals", "locals", "vars", "dir", "help", "breakpoint", "__import__"}
try:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal is disabled")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes are disabled")
        if isinstance(node, ast.Name) and node.id in banned_names:
            raise ValueError(f"name is disabled: {node.id}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            if any(name not in allowed_modules for name in names):
                raise ValueError(f"module is disabled: {names}")
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    safe_names = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError"]
    safe_builtins = {name: getattr(builtins, name) for name in safe_names}
    real_import = builtins.__import__
    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in allowed_modules:
            raise ImportError(f"module is disabled: {name}")
        return real_import(name, globals, locals, fromlist, level)
    safe_builtins["__import__"] = limited_import
    scope = {"__builtins__": safe_builtins}
    exec(compile(tree, "<tool>", "exec"), scope, scope)
except BaseException as exc:
    print("__TOOL_ERROR__" + json.dumps({"type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)
'''


def execute_python(code: str, *, timeout_seconds: float = 3.0, max_output_chars: int = 8_000) -> PythonExecResult:
    if not code.strip():
        return PythonExecResult(False, "PYTHON_ERROR: empty program", "EmptyProgram", None)
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _WRAPPER],
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PythonExecResult(False, "PYTHON_ERROR: TimeoutExpired", "TimeoutExpired", None)
    stdout = process.stdout[:max_output_chars]
    if process.returncode == 0:
        return PythonExecResult(True, stdout.rstrip() or "<no stdout>", None, 0)
    error_type = "ExecutionError"
    message = process.stderr.strip()
    marker = "__TOOL_ERROR__"
    if marker in message:
        try:
            payload = json.loads(message.split(marker, 1)[1].splitlines()[0])
            error_type = str(payload.get("type", error_type))
            message = str(payload.get("message", message))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    output = f"PYTHON_ERROR[{error_type}]: {message}"[:max_output_chars]
    return PythonExecResult(False, output, error_type, process.returncode)
