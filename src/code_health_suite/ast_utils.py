#!/usr/bin/env python3
"""ast_utils — Safe AST traversal utilities for Python analysis tools.

Prevents the ast.walk nested scope leakage bug pattern where ast.walk()
traverses into nested function/class bodies, causing incorrect scope
attribution in analysis tools.

Known bugs caused by this pattern:
  - BUG-40 (ai-complexity): compute_cyclomatic counted nested function CC
  - BUG-41 (ai-dead-code): find_unused_variables attributed inner vars to outer

Usage:
    from ast_utils import walk_scope

    # Instead of:  for child in ast.walk(function_node): ...
    # Use:         for child in walk_scope(function_node): ...

    # The boundary nodes (nested def/class) are yielded,
    # but their children are NOT traversed.
"""
from __future__ import annotations

import ast
from typing import Iterator


__version__ = "0.1.0"

# Node types that create a new scope boundary.
# When encountered during traversal, we yield the node itself
# (so callers can see it exists) but do NOT descend into its body.
SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def walk_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Walk AST nodes within the current scope, skipping nested scope bodies.

    Like ``ast.walk()`` but stops at scope boundaries (nested function/class
    definitions). The boundary node itself IS yielded so callers can detect
    its presence, but its children are NOT traversed.

    This is the correct traversal for any function-level analysis that should
    not "leak" into nested definitions — cyclomatic complexity, variable
    assignment tracking, control-flow analysis, etc.

    Args:
        node: The root AST node (typically a FunctionDef or Module).

    Yields:
        Child AST nodes within the same scope.

    Example::

        def outer():           # root node
            x = 1              # ✓ yielded
            if cond:           # ✓ yielded
                y = 2          # ✓ yielded
            def inner():       # ✓ yielded (boundary node)
                z = 3          # ✗ NOT yielded (nested scope)
            class Nested:      # ✓ yielded (boundary node)
                attr = 4       # ✗ NOT yielded (nested scope)
    """
    # DFS using a list-based stack. We push all immediate children of
    # non-boundary nodes. Boundary nodes are yielded but not expanded.
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, SCOPE_BOUNDARIES):
            stack.extend(ast.iter_child_nodes(child))


def walk_scope_bfs(node: ast.AST) -> Iterator[ast.AST]:
    """BFS variant of walk_scope — yields nodes in breadth-first order.

    Same scope-boundary semantics as ``walk_scope()``, but uses BFS instead
    of DFS. Useful when processing order matters (e.g., you want to see
    all top-level statements before nested ones).
    """
    from collections import deque
    queue = deque(ast.iter_child_nodes(node))
    while queue:
        child = queue.popleft()
        yield child
        if not isinstance(child, SCOPE_BOUNDARIES):
            queue.extend(ast.iter_child_nodes(child))


def collect_scope_names(
    node: ast.AST,
    *,
    assignments: bool = True,
    reads: bool = False,
) -> dict[str, list[int]]:
    """Collect variable names within the current scope only.

    Args:
        node: Root AST node to analyze.
        assignments: Include assignment targets (Name in Store context).
        reads: Include name reads (Name in Load context).

    Returns:
        Dict mapping variable name to list of line numbers.
    """
    names: dict[str, list[int]] = {}

    for child in walk_scope(node):
        # Reads: Name in Load context
        if reads and isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.setdefault(child.id, []).append(child.lineno)

        # Assignments: extract from specific assignment node types
        # (NOT from bare Name/Store, which includes annotation-only targets)
        if assignments:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    _collect_assign_targets(target, names)
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    names.setdefault(child.target.id, []).append(child.lineno)
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                if isinstance(child.target, ast.Name):
                    names.setdefault(child.target.id, []).append(child.lineno)
            elif isinstance(child, ast.NamedExpr):
                # Walrus operator: (x := value)
                names.setdefault(child.target.id, []).append(child.lineno)
            elif isinstance(child, ast.For) or isinstance(child, ast.AsyncFor):
                _collect_assign_targets(child.target, names)
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if item.optional_vars is not None:
                        _collect_assign_targets(item.optional_vars, names)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.setdefault(child.name, []).append(child.lineno)

    return names


def _collect_assign_targets(
    target: ast.AST, names: dict[str, list[int]]
) -> None:
    """Recursively collect Name nodes from assignment targets."""
    if isinstance(target, ast.Name):
        names.setdefault(target.id, []).append(target.lineno)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_assign_targets(elt, names)
    elif isinstance(target, ast.Starred):
        # Handle starred unpacking: a, *rest, z = items
        _collect_assign_targets(target.value, names)


def count_scope_incrementors(
    node: ast.AST,
    incrementors: tuple,
) -> int:
    """Count occurrences of specific node types within current scope.

    Useful for cyclomatic complexity, where you count decision points
    only within the function's own scope (not nested functions).

    Args:
        node: Root AST node.
        incrementors: Tuple of AST node types to count.

    Returns:
        Number of matching nodes found in scope.
    """
    count = 0
    for child in walk_scope(node):
        if isinstance(child, incrementors):
            count += 1
    return count


def is_scope_boundary(node: ast.AST) -> bool:
    """Check if a node creates a new scope boundary."""
    return isinstance(node, SCOPE_BOUNDARIES)
