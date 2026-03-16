#!/usr/bin/env python3
"""Bug detection engine: finds common Python semantic bugs via AST analysis.

Detects 8 categories of real bugs (not style issues):
1. missing-fstring: strings with {var} that forgot the f-prefix
2. mutable-class-var: class-level mutable defaults shared across instances
3. late-binding-closure: closures in loops/comprehensions capturing loop var
4. call-default: call expressions (datetime.now()) evaluated once at def time
5. mutable-default-arg: mutable default arguments shared across calls
6. assert-tuple: assert(cond, msg) which is always True (tuple is truthy)
7. unreachable-code: code after return/raise/break/continue
8. unreachable-except: except handlers shadowed by broader earlier handler

Zero external dependencies. Pure AST analysis.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


# --- Data classes ---

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".nox", ".eggs", "node_modules", "venv", ".venv", "env",
    ".env", "dist", "build", "egg-info", ".ruff_cache",
}


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    file: str
    line: int
    context: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileResult:
    path: str
    findings: list[Finding] = field(default_factory=list)
    error: str = ""


@dataclass
class ScanResult:
    root: str
    files_scanned: int = 0
    total_findings: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    files: list[FileResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "total_findings": self.total_findings,
            "by_rule": self.by_rule,
            "by_severity": self.by_severity,
            "findings": [
                f.to_dict()
                for fr in self.files
                for f in fr.findings
            ],
            "errors": self.errors,
        }


@dataclass
class ScoreResult:
    score: int       # 0-100
    grade: str       # A-F
    profile: str     # clean/fstring_heavy/closure_heavy/etc.
    files_scanned: int
    files_with_findings: int
    clean_file_pct: float
    total_findings: int
    density: float   # findings per file
    by_rule: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    dominant_rule: str = ""
    dominant_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --- AST helpers ---

def _add_parents(tree: ast.AST) -> None:
    """Annotate every node with a .parent reference."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _get_docstring_nodes(tree: ast.AST) -> set:
    """Return set of AST node ids that are docstrings."""
    docstring_ids: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    docstring_ids.add(id(val))
    return docstring_ids


def _get_call_name(node: ast.Call) -> str:
    """Extract dotted name from a Call node, e.g. 'datetime.now'."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return ""


def _is_in_scope(node: ast.AST, scope_types: tuple) -> bool:
    """Check if node is inside any of the given scope types."""
    current = getattr(node, "parent", None)
    while current is not None:
        if isinstance(current, scope_types):
            return True
        current = getattr(current, "parent", None)
    return False


# --- Detector 1: Missing f-string ---

_FSTRING_CANDIDATE = re.compile(
    r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\[[^\]]*\])?)\}(?!\})"
)

_TEMPLATE_NAMES = frozenset({
    "template", "pattern", "regex", "fmt", "format_str", "sql", "query",
    "url_template", "path_template", "msg_template", "tmpl", "tpl",
    "FORMAT", "PATTERN", "TEMPLATE", "SQL", "QUERY",
})

_TEMPLATE_KWARGS = frozenset({
    "help", "metavar", "description", "epilog", "usage", "prog",
    "template", "format", "fmt", "pattern", "url", "endpoint",
})


def detect_missing_fstring(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect string literals that look like they should be f-strings."""
    findings: list[Finding] = []
    docstring_ids = _get_docstring_nodes(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        if id(node) in docstring_ids:
            continue

        s = node.value
        if not _FSTRING_CANDIDATE.search(s):
            continue

        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Attribute) and parent.attr == "format":
            continue
        if isinstance(parent, ast.Attribute):
            grandparent = getattr(parent, "parent", None)
            if isinstance(grandparent, ast.Call) and grandparent.func is parent:
                continue

        if isinstance(parent, ast.Assign):
            _is_template_assign = False
            for target in parent.targets:
                if isinstance(target, ast.Name):
                    if target.id in _TEMPLATE_NAMES:
                        _is_template_assign = True
                        break
                    if target.id.isupper() or (target.id.startswith("_") and target.id[1:].isupper()):
                        _is_template_assign = True
                        break
            if _is_template_assign:
                continue

        if isinstance(parent, ast.keyword) and parent.arg in _TEMPLATE_KWARGS:
            continue

        _ancestor = parent
        _is_template_call = False
        for _ in range(4):
            if _ancestor is None:
                break
            if isinstance(_ancestor, ast.Call):
                call_name = _get_call_name(_ancestor)
                if call_name and any(t in call_name.lower() for t in (
                    "column", "template", "pattern", "prompt", "message",
                )):
                    _is_template_call = True
                    break
            _ancestor = getattr(_ancestor, "parent", None)
        if _is_template_call:
            continue

        if isinstance(parent, ast.Attribute) and parent.attr == "replace":
            continue
        if "{{" in s:
            continue
        if "${" in s or "%{" in s:
            continue
        if re.match(r'^\{[a-zA-Z_][a-zA-Z0-9_.]*\}$', s):
            continue
        if s.count("\n") >= 3:
            continue
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mod):
            continue

        matches = _FSTRING_CANDIDATE.findall(s)
        if matches:
            preview = s[:60] + "..." if len(s) > 60 else s
            findings.append(Finding(
                rule="missing-fstring",
                severity="warning",
                message="String contains {...} placeholders but is not an f-string",
                file=filepath,
                line=node.lineno,
                context=repr(preview),
                detail=f"Possible missing f-prefix. Placeholders: {matches[:3]}",
            ))

    return findings


# --- Detector 2: Mutable class variable ---

_MUTABLE_TYPES = (ast.List, ast.Dict, ast.Set)

_FRAMEWORK_CLASS_VARS = frozenset({
    "minimal_roi", "stoploss", "trailing_stop", "trailing_stop_positive",
    "trailing_stop_positive_offset", "order_types", "order_time_in_force",
    "protections", "pair_blacklist", "pair_whitelist",
    "fields", "exclude", "widgets", "labels", "help_texts",
    "error_messages", "field_classes",
    "__table_args__",
    "pytestmark",
})


def detect_mutable_class_var(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect mutable default values in class body (shared across instances)."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and isinstance(stmt.value, _MUTABLE_TYPES):
                        if target.id == "__all__":
                            continue
                        if target.id.isupper() or (target.id.startswith("_") and target.id[1:].isupper()):
                            continue
                        if target.id in _FRAMEWORK_CLASS_VARS:
                            continue
                        mtype = type(stmt.value).__name__.lower()
                        findings.append(Finding(
                            rule="mutable-class-var",
                            severity="warning",
                            message=f"Mutable {mtype} as class variable '{target.id}' is shared across all instances",
                            file=filepath,
                            line=stmt.lineno,
                            context=f"class {node.name}: {target.id} = {mtype}()",
                            detail="Move to __init__ or use dataclass field(default_factory=...)",
                        ))

            if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, _MUTABLE_TYPES):
                    name = stmt.target.id
                    if name == "__all__":
                        continue
                    if name.isupper() or (name.startswith("_") and name[1:].isupper()):
                        continue
                    if name in _FRAMEWORK_CLASS_VARS:
                        continue
                    mtype = type(stmt.value).__name__.lower()
                    findings.append(Finding(
                        rule="mutable-class-var",
                        severity="warning",
                        message=f"Mutable {mtype} as class variable '{name}' is shared across all instances",
                        file=filepath,
                        line=stmt.lineno,
                        context=f"class {node.name}: {name}: ... = {mtype}()",
                        detail="Move to __init__ or use dataclass field(default_factory=...)",
                    ))

    return findings


# --- Detector 3: Late-binding closure ---

_IMMEDIATE_CONSUMERS = frozenset({
    "max", "min", "sorted", "filter", "map", "reduce",
    "any", "all", "next",
})


def _get_loop_vars(node: ast.AST) -> set:
    """Extract variable names from for-loop targets."""
    names: set = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(_get_loop_vars(elt))
    return names


def _references_names(node: ast.AST, names: set) -> set:
    """Return which names from `names` are referenced in node's subtree."""
    found: set = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            found.add(child.id)
    return found


def _is_immediate_lambda_use(loop_node: ast.AST, lambda_node: ast.Lambda) -> bool:
    """Check if a lambda is used as an immediate argument to a function call."""
    for node in ast.walk(loop_node):
        if not isinstance(node, ast.Call):
            continue
        if any(kw.value is lambda_node for kw in node.keywords):
            return True
        if lambda_node in node.args:
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name in _IMMEDIATE_CONSUMERS:
                return True
    return False


def detect_late_binding_closure(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect closures in loops/comprehensions that capture loop variable."""
    findings: list[Finding] = []

    # For loops
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue

        loop_vars = _get_loop_vars(node.target)
        if not loop_vars:
            continue

        for child in ast.walk(node):
            if child is node:
                continue

            if isinstance(child, ast.Lambda):
                lambda_params = {arg.arg for arg in child.args.args}
                lambda_params.update(arg.arg for arg in getattr(child.args, "posonlyargs", []))
                lambda_params.update(arg.arg for arg in child.args.kwonlyargs)
                if child.args.vararg:
                    lambda_params.add(child.args.vararg.arg)
                if child.args.kwarg:
                    lambda_params.add(child.args.kwarg.arg)

                captured = _references_names(child.body, loop_vars)
                captured -= lambda_params
                for i, d in enumerate(child.args.defaults):
                    idx = len(child.args.args) - len(child.args.defaults) + i
                    if idx >= 0 and child.args.args[idx].arg in captured:
                        captured.discard(child.args.args[idx].arg)

                if captured and not _is_immediate_lambda_use(node, child):
                    findings.append(Finding(
                        rule="late-binding-closure",
                        severity="warning",
                        message=f"Lambda in loop captures variable '{', '.join(sorted(captured))}' by reference",
                        file=filepath,
                        line=child.lineno,
                        context=f"for {ast.dump(node.target)}:  lambda: ...{', '.join(sorted(captured))}...",
                        detail="All closures will see the last value. Use default arg: lambda x=x: ...",
                    ))

            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_names = {a.arg for a in child.args.args}
                param_names.update(a.arg for a in child.args.posonlyargs)
                param_names.update(a.arg for a in child.args.kwonlyargs)
                if child.args.vararg:
                    param_names.add(child.args.vararg.arg)
                if child.args.kwarg:
                    param_names.add(child.args.kwarg.arg)

                captured = set()
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Name) and sub.id in loop_vars and sub.id not in param_names:
                        captured.add(sub.id)

                if captured:
                    findings.append(Finding(
                        rule="late-binding-closure",
                        severity="warning",
                        message=f"Function '{child.name}' in loop captures variable '{', '.join(sorted(captured))}' by reference",
                        file=filepath,
                        line=child.lineno,
                        context=f"for ...: def {child.name}(): ...{', '.join(sorted(captured))}...",
                        detail="All closures will see the last value. Pass as default parameter.",
                    ))

    # Comprehension closures
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue

        comp_vars: set = set()
        for gen in node.generators:
            comp_vars.update(_get_loop_vars(gen.target))

        if not comp_vars:
            continue

        if isinstance(node, ast.DictComp):
            search_nodes = [node.key, node.value]
        else:
            search_nodes = [node.elt]

        for search_root in search_nodes:
            for child in ast.walk(search_root):
                if isinstance(child, ast.Lambda):
                    lambda_params = {arg.arg for arg in child.args.args}
                    lambda_params.update(arg.arg for arg in getattr(child.args, "posonlyargs", []))
                    lambda_params.update(arg.arg for arg in child.args.kwonlyargs)
                    if child.args.vararg:
                        lambda_params.add(child.args.vararg.arg)
                    if child.args.kwarg:
                        lambda_params.add(child.args.kwarg.arg)

                    captured = _references_names(child.body, comp_vars)
                    captured -= lambda_params
                    for i, d in enumerate(child.args.defaults):
                        idx = len(child.args.args) - len(child.args.defaults) + i
                        if idx >= 0 and child.args.args[idx].arg in captured:
                            captured.discard(child.args.args[idx].arg)

                    if captured and not _is_immediate_lambda_use(search_root, child):
                        findings.append(Finding(
                            rule="late-binding-closure",
                            severity="warning",
                            message=f"Lambda in comprehension captures variable '{', '.join(sorted(captured))}' by reference",
                            file=filepath,
                            line=child.lineno,
                            context=f"[lambda: ...{', '.join(sorted(captured))}... for ...]",
                            detail="All closures will see the last value. Use default arg: lambda x=x: ...",
                        ))

                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    param_names = {a.arg for a in child.args.args}
                    param_names.update(a.arg for a in child.args.posonlyargs)
                    param_names.update(a.arg for a in child.args.kwonlyargs)
                    if child.args.vararg:
                        param_names.add(child.args.vararg.arg)
                    if child.args.kwarg:
                        param_names.add(child.args.kwarg.arg)

                    captured = set()
                    for sub in ast.walk(child):
                        if isinstance(sub, ast.Name) and sub.id in comp_vars and sub.id not in param_names:
                            captured.add(sub.id)

                    if captured:
                        findings.append(Finding(
                            rule="late-binding-closure",
                            severity="warning",
                            message=f"Function '{child.name}' in comprehension captures variable '{', '.join(sorted(captured))}' by reference",
                            file=filepath,
                            line=child.lineno,
                            context=f"[def {child.name}(): ...{', '.join(sorted(captured))}... for ...]",
                            detail="All closures will see the last value. Pass as default parameter.",
                        ))

    return findings


# --- Detector 4: Call expression as default ---

_DANGEROUS_DEFAULTS = frozenset({
    "datetime.now", "datetime.utcnow", "datetime.today",
    "date.today", "time.time", "time.monotonic", "time.perf_counter",
    "uuid.uuid4", "uuid.uuid1",
    "random.random", "random.randint", "random.choice",
    "os.getenv", "os.environ.get",
})

_DANGEROUS_DEFAULT_SHORT = frozenset({
    "now", "utcnow", "today", "uuid4", "uuid1",
})


def detect_call_default(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect function calls as default argument values."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        all_defaults = list(node.args.defaults) + list(node.args.kw_defaults)

        for default in all_defaults:
            if default is None:
                continue
            if not isinstance(default, ast.Call):
                continue

            call_name = _get_call_name(default)
            if call_name in _DANGEROUS_DEFAULTS or call_name in _DANGEROUS_DEFAULT_SHORT:
                findings.append(Finding(
                    rule="call-default",
                    severity="warning",
                    message=f"Call '{call_name}()' as default argument is evaluated once at definition time",
                    file=filepath,
                    line=default.lineno,
                    context=f"def {node.name}(..., x={call_name}())",
                    detail="Use None as default and set inside function body.",
                ))

    return findings


# --- Detector 5: Mutable default argument ---

_MUTABLE_CONSTRUCTORS = frozenset({
    "list", "dict", "set", "bytearray",
    "deque", "defaultdict", "OrderedDict", "Counter",
})


def detect_mutable_default_arg(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect mutable default arguments in function definitions."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        all_defaults = list(node.args.defaults) + list(node.args.kw_defaults)

        for default in all_defaults:
            if default is None:
                continue

            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                mtype = type(default).__name__.lower()
                findings.append(Finding(
                    rule="mutable-default-arg",
                    severity="warning",
                    message=f"Mutable {mtype} as default argument is shared across all calls",
                    file=filepath,
                    line=default.lineno,
                    context=f"def {node.name}(..., x={mtype}(...))",
                    detail="Use None as default: def f(x=None): x = x if x is not None else []",
                ))
                continue

            if isinstance(default, ast.Call):
                call_name = _get_call_name(default)
                if call_name in _MUTABLE_CONSTRUCTORS:
                    findings.append(Finding(
                        rule="mutable-default-arg",
                        severity="warning",
                        message=f"Mutable {call_name}() as default argument is shared across all calls",
                        file=filepath,
                        line=default.lineno,
                        context=f"def {node.name}(..., x={call_name}())",
                        detail="Use None as default: def f(x=None): x = x if x is not None else []",
                    ))

    return findings


# --- Detector 6: Assert tuple ---

def detect_assert_tuple(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect `assert(condition, message)` which is always True."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue

        if isinstance(node.test, ast.Tuple) and len(node.test.elts) >= 2:
            findings.append(Finding(
                rule="assert-tuple",
                severity="error",
                message="assert with tuple is always True — tuple is truthy",
                file=filepath,
                line=node.lineno,
                context="assert(cond, msg)  # always True",
                detail="Remove parentheses: use `assert cond, msg` not `assert(cond, msg)`",
            ))

    return findings


# --- Detector 7: Unreachable code ---

_TERMINAL_TYPES = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def detect_unreachable_code(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect code after return/raise/break/continue in the same block."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(node, attr, None)
            if not isinstance(stmts, list):
                continue

            stmt_list = [s for s in stmts if isinstance(s, ast.AST) and hasattr(s, "lineno")]
            for i, stmt in enumerate(stmt_list):
                if isinstance(stmt, _TERMINAL_TYPES) and i < len(stmt_list) - 1:
                    next_stmt = stmt_list[i + 1]
                    if isinstance(next_stmt, ast.Expr) and isinstance(getattr(next_stmt, "value", None), ast.Constant):
                        if isinstance(next_stmt.value.value, str):
                            continue
                    terminal_name = type(stmt).__name__.lower()
                    findings.append(Finding(
                        rule="unreachable-code",
                        severity="warning",
                        message=f"Code after '{terminal_name}' is unreachable",
                        file=filepath,
                        line=next_stmt.lineno,
                        context=f"Line {stmt.lineno}: {terminal_name} → line {next_stmt.lineno}: unreachable",
                        detail="This code will never execute. Remove it or fix the control flow.",
                    ))
                    break

    return findings


# --- Detector 8: Unreachable except handler ---

_EXCEPTION_PARENTS = {
    "BaseException": None,
    "Exception": "BaseException",
    "GeneratorExit": "BaseException",
    "KeyboardInterrupt": "BaseException",
    "SystemExit": "BaseException",
    "ArithmeticError": "Exception",
    "FloatingPointError": "ArithmeticError",
    "OverflowError": "ArithmeticError",
    "ZeroDivisionError": "ArithmeticError",
    "LookupError": "Exception",
    "IndexError": "LookupError",
    "KeyError": "LookupError",
    "NameError": "Exception",
    "UnboundLocalError": "NameError",
    "OSError": "Exception",
    "IOError": "OSError",
    "EnvironmentError": "OSError",
    "BlockingIOError": "OSError",
    "ChildProcessError": "OSError",
    "ConnectionError": "OSError",
    "BrokenPipeError": "ConnectionError",
    "ConnectionAbortedError": "ConnectionError",
    "ConnectionRefusedError": "ConnectionError",
    "ConnectionResetError": "ConnectionError",
    "FileExistsError": "OSError",
    "FileNotFoundError": "OSError",
    "InterruptedError": "OSError",
    "IsADirectoryError": "OSError",
    "NotADirectoryError": "OSError",
    "PermissionError": "OSError",
    "ProcessLookupError": "OSError",
    "TimeoutError": "OSError",
    "RuntimeError": "Exception",
    "NotImplementedError": "RuntimeError",
    "RecursionError": "RuntimeError",
    "SyntaxError": "Exception",
    "IndentationError": "SyntaxError",
    "TabError": "IndentationError",
    "ValueError": "Exception",
    "UnicodeError": "ValueError",
    "UnicodeDecodeError": "UnicodeError",
    "UnicodeEncodeError": "UnicodeError",
    "UnicodeTranslateError": "UnicodeError",
    "ImportError": "Exception",
    "ModuleNotFoundError": "ImportError",
    "AssertionError": "Exception",
    "AttributeError": "Exception",
    "BufferError": "Exception",
    "EOFError": "Exception",
    "MemoryError": "Exception",
    "ReferenceError": "Exception",
    "StopIteration": "Exception",
    "StopAsyncIteration": "Exception",
    "SystemError": "Exception",
    "TypeError": "Exception",
}


def _is_builtin_exception_subclass(child: str, parent: str) -> bool:
    """Check if child is a subclass of parent using builtin exception hierarchy."""
    if child == parent:
        return True
    current = _EXCEPTION_PARENTS.get(child)
    while current is not None:
        if current == parent:
            return True
        current = _EXCEPTION_PARENTS.get(current)
    return False


def _get_handler_exception_names(handler: ast.ExceptHandler) -> list[str]:
    """Extract exception class names from an except handler."""
    if handler.type is None:
        return ["BaseException"]
    if isinstance(handler.type, ast.Name):
        return [handler.type.id]
    if isinstance(handler.type, ast.Tuple):
        return [elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)]
    return []


def detect_unreachable_except(tree: ast.AST, filepath: str) -> list[Finding]:
    """Detect except handlers that can never be reached."""
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue

        seen_types: list[tuple] = []

        for i, handler in enumerate(node.handlers):
            handler_types = _get_handler_exception_names(handler)
            if not handler_types:
                continue

            all_covered = True
            covering_type = None
            covering_line = None
            for htype in handler_types:
                covered = False
                for _, prev_type, prev_line in seen_types:
                    if (htype in _EXCEPTION_PARENTS and prev_type in _EXCEPTION_PARENTS
                            and _is_builtin_exception_subclass(htype, prev_type)):
                        covered = True
                        covering_type = prev_type
                        covering_line = prev_line
                        break
                if not covered:
                    all_covered = False
                    break

            if all_covered and handler_types:
                types_str = ", ".join(handler_types)
                findings.append(Finding(
                    rule="unreachable-except",
                    severity="error",
                    message=f"except {types_str} is unreachable — already caught by {covering_type} (line {covering_line})",
                    file=filepath,
                    line=handler.lineno,
                    context=f"Handler #{i+1} in try block at line {node.lineno}",
                    detail=f"{covering_type} on line {covering_line} catches {types_str}, making this handler dead code",
                ))

            for htype in handler_types:
                seen_types.append((i, htype, handler.lineno))

    return findings


# --- Analyzer ---

ALL_DETECTORS = [
    detect_missing_fstring,
    detect_mutable_class_var,
    detect_mutable_default_arg,
    detect_late_binding_closure,
    detect_call_default,
    detect_assert_tuple,
    detect_unreachable_code,
    detect_unreachable_except,
]


def analyze_file(filepath: str) -> FileResult:
    """Run all detectors on a single Python file."""
    result = FileResult(path=filepath)
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        result.error = str(e)
        return result

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        result.error = f"SyntaxError: {e}"
        return result

    _add_parents(tree)

    for detector in ALL_DETECTORS:
        result.findings.extend(detector(tree, filepath))

    result.findings.sort(key=lambda f: (f.line, SEVERITY_ORDER.get(f.severity, 9)))
    return result


def find_python_files(path: str) -> list[str]:
    """Find Python files under a path, skipping non-source dirs."""
    if os.path.isfile(path):
        return [path] if path.endswith(".py") else []

    files = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))
    return files


def scan(path: str, rules: list[str] | None = None, min_severity: str = "info") -> ScanResult:
    """Scan a file or directory for bugs. Main entry point."""
    result = ScanResult(root=path)
    py_files = find_python_files(path)

    min_sev = SEVERITY_ORDER.get(min_severity, 2)

    for filepath in py_files:
        fr = analyze_file(filepath)
        if fr.error:
            result.errors.append(f"{filepath}: {fr.error}")
            continue

        # Filter by severity
        fr.findings = [f for f in fr.findings if SEVERITY_ORDER.get(f.severity, 9) <= min_sev]

        # Filter by rules
        if rules:
            fr.findings = [f for f in fr.findings if f.rule in rules]

        result.files.append(fr)
        result.files_scanned += 1

        for f in fr.findings:
            result.total_findings += 1
            result.by_rule[f.rule] = result.by_rule.get(f.rule, 0) + 1
            result.by_severity[f.severity] = result.by_severity.get(f.severity, 0) + 1

    return result


# --- Scoring ---

_SEVERITY_WEIGHTS = {"error": 5.0, "warning": 2.0, "info": 0.5}


def _score_to_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _classify_profile(by_rule: dict, total_findings: int) -> tuple[str, str, float]:
    """Classify bug profile. Returns (profile, dominant_rule, dominant_pct)."""
    if total_findings <= 1:
        dominant = max(by_rule, key=by_rule.get) if by_rule else "none"
        dominant_pct = (by_rule.get(dominant, 0) / max(total_findings, 1)) * 100
        return "clean", dominant, dominant_pct

    dominant = max(by_rule, key=by_rule.get) if by_rule else "none"
    dominant_pct = (by_rule.get(dominant, 0) / total_findings) * 100

    control_flow = by_rule.get("unreachable-code", 0) + by_rule.get("unreachable-except", 0)
    control_flow_pct = (control_flow / total_findings) * 100

    profile_map = {
        "missing-fstring": "fstring_heavy",
        "late-binding-closure": "closure_heavy",
        "mutable-class-var": "class_var_heavy",
        "mutable-default-arg": "default_heavy",
        "call-default": "default_heavy",
        "assert-tuple": "mixed",
    }

    if control_flow_pct > 50:
        return "control_flow_heavy", dominant, dominant_pct
    if dominant_pct > 50:
        return profile_map.get(dominant, "mixed"), dominant, dominant_pct
    return "mixed", dominant, dominant_pct


def compute_score(result: ScanResult) -> ScoreResult:
    """Compute project-level bug detection health score (0-100)."""
    files_scanned = max(result.files_scanned, 1)
    files_with_findings = sum(1 for fr in result.files if fr.findings)
    clean_pct = ((files_scanned - files_with_findings) / files_scanned) * 100
    density = result.total_findings / files_scanned

    weighted_sum = sum(
        result.by_severity.get(sev, 0) * w
        for sev, w in _SEVERITY_WEIGHTS.items()
    )
    weighted_density = weighted_sum / files_scanned
    penalty = min(100, weighted_density * 12)
    score = max(0, min(100, round(100 - penalty)))

    grade = _score_to_grade(score)
    profile, dominant, dominant_pct = _classify_profile(
        result.by_rule, result.total_findings
    )

    return ScoreResult(
        score=score,
        grade=grade,
        profile=profile,
        files_scanned=files_scanned,
        files_with_findings=files_with_findings,
        clean_file_pct=round(clean_pct, 1),
        total_findings=result.total_findings,
        density=round(density, 2),
        by_rule=dict(result.by_rule),
        by_severity=dict(result.by_severity),
        dominant_rule=dominant,
        dominant_pct=round(dominant_pct, 1),
    )
