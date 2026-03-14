#!/usr/bin/env python3
"""ai-clone-detect: AST-based Python code clone detector.

Detects duplicated and similar code blocks across Python files.
Supports Type-1 (exact), Type-2 (renamed), and Type-3 (modified) clones.
Zero external dependencies.

Usage:
    ai-clone-detect                           # scan current directory
    ai-clone-detect path/to/project           # scan specific directory
    ai-clone-detect -f file1.py file2.py      # scan specific files
    ai-clone-detect --threshold 0.9           # stricter similarity
    ai-clone-detect --min-lines 10            # larger minimum block size
    ai-clone-detect --json                    # JSON output
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


__version__ = "0.2.0"

# --- Defaults ---

DEFAULT_THRESHOLD = 0.8
DEFAULT_MIN_LINES = 5
DEFAULT_MAX_FILES = 500


# --- Data classes ---

@dataclass
class CodeBlock:
    """A function, method, or class extracted from source."""
    name: str
    filepath: str
    start_line: int
    end_line: int
    source: str
    normalized: str
    block_type: str  # "function", "method", "class"
    node_count: int

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class ClonePair:
    """A detected pair of similar code blocks."""
    block_a: CodeBlock
    block_b: CodeBlock
    similarity: float
    clone_type: str  # "type-1", "type-2", "type-3"


@dataclass
class CloneCluster:
    """A group of code blocks that are all clones of each other."""
    cluster_id: int
    blocks: list[CodeBlock]
    max_similarity: float
    min_similarity: float
    dominant_type: str  # most severe clone type in the cluster

    @property
    def size(self) -> int:
        return len(self.blocks)

    @property
    def total_duplicated_lines(self) -> int:
        """Lines that could be saved by extracting to a shared function."""
        if self.size <= 1:
            return 0
        # Keep one copy, all others are "duplicated"
        line_counts = sorted((b.line_count for b in self.blocks), reverse=True)
        return sum(line_counts[1:])


@dataclass
class ScanResult:
    """Summary of a clone detection scan."""
    files_scanned: int
    blocks_extracted: int
    clone_pairs: list[ClonePair]
    errors: list[dict]
    clusters: list[CloneCluster] = None  # type: ignore[assignment]
    clone_score: int = 100  # 0-100, higher is better (fewer clones)


# --- AST normalization ---

def _count_nodes(node: ast.AST) -> int:
    """Count total AST nodes in a subtree."""
    count = 1
    for child in ast.iter_child_nodes(node):
        count += _count_nodes(child)
    return count


class _Normalizer(ast.NodeTransformer):
    """Replace all identifiers with sequential placeholders for structural comparison."""

    def __init__(self):
        self._name_map: dict[str, str] = {}
        self._counter = 0

    def _get_placeholder(self, name: str) -> str:
        if name not in self._name_map:
            self._name_map[name] = f"V{self._counter}"
            self._counter += 1
        return self._name_map[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._get_placeholder(node.id)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._get_placeholder(node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.name = self._get_placeholder(node.name)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._get_placeholder(node.arg)
        node.annotation = None
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.name = self._get_placeholder(node.name)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        # Keep string/number constants for type-1 matching but normalize for type-2
        if isinstance(node.value, str):
            node.value = "S"
        elif isinstance(node.value, (int, float, complex)):
            node.value = 0
        return node

    def visit_alias(self, node: ast.alias) -> ast.alias:
        node.name = self._get_placeholder(node.name)
        if node.asname:
            node.asname = self._get_placeholder(node.asname)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        if node.arg:
            node.arg = self._get_placeholder(node.arg)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        node.attr = self._get_placeholder(node.attr)
        return self.generic_visit(node)


def normalize_ast(node: ast.AST) -> str:
    """Normalize an AST node by replacing all identifiers and return canonical string."""
    import copy
    clone = copy.deepcopy(node)
    normalizer = _Normalizer()
    normalized = normalizer.visit(clone)
    # Strip line numbers and col offsets for structural comparison
    for n in ast.walk(normalized):
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if hasattr(n, attr):
                setattr(n, attr, 0)
    return ast.dump(normalized)


def normalize_source(node: ast.AST) -> str:
    """Return a normalized source-like string for display purposes."""
    import copy
    clone = copy.deepcopy(node)
    normalizer = _Normalizer()
    normalized = normalizer.visit(clone)
    try:
        return ast.unparse(normalized)
    except Exception:
        return ast.dump(normalized)


# --- Block extraction ---

def extract_blocks(source: str, filepath: str, min_lines: int = DEFAULT_MIN_LINES) -> list[CodeBlock]:
    """Extract function, method, and class blocks from Python source code."""
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_func(node, filepath, min_lines, blocks)
        elif isinstance(node, ast.ClassDef):
            _extract_class(node, filepath, min_lines, blocks)
    return blocks


def _extract_func(node, filepath: str, min_lines: int, blocks: list[CodeBlock]):
    """Extract a function/method block."""
    start = node.lineno
    end = node.end_lineno or start
    if end - start + 1 < min_lines:
        return

    # Determine if method (inside a class)
    block_type = "function"
    # We can't easily know parent from ast.walk, but name convention hints
    # Methods are extracted when walking class bodies too, so this is fine

    try:
        source = ast.unparse(node)
    except Exception:
        return

    normalized = normalize_ast(node)
    node_count = _count_nodes(node)

    blocks.append(CodeBlock(
        name=node.name,
        filepath=filepath,
        start_line=start,
        end_line=end,
        source=source,
        normalized=normalized,
        block_type=block_type,
        node_count=node_count,
    ))


def _extract_class(node: ast.ClassDef, filepath: str, min_lines: int, blocks: list[CodeBlock]):
    """Extract a class block (the class itself, not its methods — those are extracted separately)."""
    start = node.lineno
    end = node.end_lineno or start
    if end - start + 1 < min_lines:
        return

    # Only extract classes with substantial non-method body
    non_method_stmts = [
        s for s in node.body
        if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(non_method_stmts) < 2:
        return

    try:
        source = ast.unparse(node)
    except Exception:
        return

    normalized = normalize_ast(node)
    node_count = _count_nodes(node)

    blocks.append(CodeBlock(
        name=node.name,
        filepath=filepath,
        start_line=start,
        end_line=end,
        source=source,
        normalized=normalized,
        block_type="class",
        node_count=node_count,
    ))


# --- Similarity computation ---

def compute_similarity(block_a: CodeBlock, block_b: CodeBlock) -> float:
    """Compute structural similarity between two code blocks (0.0 to 1.0)."""
    # Quick reject: if node counts differ by more than 50%, skip detailed comparison
    if block_a.node_count == 0 or block_b.node_count == 0:
        return 0.0
    ratio = min(block_a.node_count, block_b.node_count) / max(block_a.node_count, block_b.node_count)
    if ratio < 0.5:
        return 0.0

    # Use SequenceMatcher on normalized AST dump strings
    return difflib.SequenceMatcher(
        None, block_a.normalized, block_b.normalized
    ).ratio()


def classify_clone(block_a: CodeBlock, block_b: CodeBlock, similarity: float) -> str:
    """Classify clone type based on similarity analysis."""
    if block_a.normalized == block_b.normalized:
        # Structurally identical after normalization
        if block_a.source == block_b.source:
            return "type-1"  # Exact copy
        return "type-2"  # Renamed identifiers
    return "type-3"  # Modified statements


# --- Clone detection ---

def find_clones(
    blocks: list[CodeBlock],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[ClonePair]:
    """Find all clone pairs above the similarity threshold."""
    clones = []
    n = len(blocks)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = blocks[i], blocks[j]

            # Skip self-comparisons within same file at same location
            if a.filepath == b.filepath and a.start_line == b.start_line:
                continue

            # Skip if blocks are nested (one contains the other in same file)
            if a.filepath == b.filepath:
                if (a.start_line <= b.start_line and a.end_line >= b.end_line):
                    continue
                if (b.start_line <= a.start_line and b.end_line >= a.end_line):
                    continue

            similarity = compute_similarity(a, b)
            if similarity >= threshold:
                clone_type = classify_clone(a, b, similarity)
                clones.append(ClonePair(
                    block_a=a,
                    block_b=b,
                    similarity=similarity,
                    clone_type=clone_type,
                ))

    # Sort by similarity descending
    clones.sort(key=lambda c: c.similarity, reverse=True)
    return clones


# --- Clustering ---

class _UnionFind:
    """Simple union-find (disjoint set) for clustering clone pairs."""

    def __init__(self):
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def cluster_clones(
    blocks: list[CodeBlock],
    clone_pairs: list[ClonePair],
) -> list[CloneCluster]:
    """Group clone pairs into clusters using union-find.

    If A≈B and B≈C, they all end up in one cluster {A, B, C}.
    """
    if not clone_pairs:
        return []

    # Build block index for O(1) lookup
    block_to_idx: dict[int, int] = {}
    for i, b in enumerate(blocks):
        block_to_idx[id(b)] = i

    uf = _UnionFind()

    # Pair metadata: track similarities and types per edge
    pair_meta: dict[tuple[int, int], tuple[float, str]] = {}
    for cp in clone_pairs:
        idx_a = block_to_idx.get(id(cp.block_a))
        idx_b = block_to_idx.get(id(cp.block_b))
        if idx_a is None or idx_b is None:
            continue
        uf.union(idx_a, idx_b)
        key = (min(idx_a, idx_b), max(idx_a, idx_b))
        pair_meta[key] = (cp.similarity, cp.clone_type)

    # Group blocks by root
    groups: dict[int, list[int]] = {}
    for cp in clone_pairs:
        for b in (cp.block_a, cp.block_b):
            idx = block_to_idx.get(id(b))
            if idx is not None:
                root = uf.find(idx)
                if root not in groups:
                    groups[root] = []
                if idx not in groups[root]:
                    groups[root].append(idx)

    # Build CloneCluster objects
    type_severity = {"type-1": 3, "type-2": 2, "type-3": 1}
    clusters = []
    for cluster_id, (_, member_idxs) in enumerate(sorted(groups.items())):
        cluster_blocks = [blocks[i] for i in sorted(member_idxs)]

        # Collect similarities and types for pairs in this cluster
        sims = []
        types = []
        for i, idx_a in enumerate(member_idxs):
            for idx_b in member_idxs[i + 1:]:
                key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if key in pair_meta:
                    sim, ctype = pair_meta[key]
                    sims.append(sim)
                    types.append(ctype)

        max_sim = max(sims) if sims else 0.0
        min_sim = min(sims) if sims else 0.0
        dominant = max(types, key=lambda t: type_severity.get(t, 0)) if types else "type-3"

        clusters.append(CloneCluster(
            cluster_id=cluster_id,
            blocks=cluster_blocks,
            max_similarity=max_sim,
            min_similarity=min_sim,
            dominant_type=dominant,
        ))

    # Sort by total duplicated lines descending (most impactful first)
    clusters.sort(key=lambda c: c.total_duplicated_lines, reverse=True)
    return clusters


def compute_clone_score(blocks_extracted: int, clusters: list[CloneCluster]) -> int:
    """Compute a clone health score (0-100, higher = fewer clones).

    Score formula:
    - Start at 100
    - Each duplicated line costs 1 point per 10 lines (scaled by severity)
    - Type-1 clones cost 3x, Type-2 cost 2x, Type-3 cost 1x
    - Floor at 0
    """
    if blocks_extracted == 0 or not clusters:
        return 100

    type_weight = {"type-1": 3.0, "type-2": 2.0, "type-3": 1.0}
    total_penalty = 0.0
    for cluster in clusters:
        weight = type_weight.get(cluster.dominant_type, 1.0)
        dup_lines = cluster.total_duplicated_lines
        total_penalty += dup_lines * weight / 10.0

    score = max(0, int(100 - total_penalty))
    return score


def score_to_grade(score: int) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


# --- File scanning ---

def collect_python_files(path: str, max_files: int = DEFAULT_MAX_FILES) -> list[str]:
    """Collect Python files from a directory, respecting limits."""
    p = Path(path)
    if p.is_file() and p.suffix == ".py":
        return [str(p)]

    SKIP_DIRS = {
        ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
        "node_modules", ".tox", ".nox", ".venv", "venv", "env",
        ".eggs", "*.egg-info", "dist", "build",
    }

    files = []
    for root, dirs, filenames in os.walk(path):
        # Skip hidden and known non-source dirs
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and not _is_virtualenv(os.path.join(root, d))
        ]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))
                if len(files) >= max_files:
                    return files
    return files


def _is_virtualenv(dirpath: str) -> bool:
    """Check if a directory is a Python virtual environment."""
    return os.path.isfile(os.path.join(dirpath, "pyvenv.cfg"))


def scan_files(
    filepaths: list[str],
    min_lines: int = DEFAULT_MIN_LINES,
    threshold: float = DEFAULT_THRESHOLD,
) -> ScanResult:
    """Scan a list of files for code clones."""
    all_blocks = []
    errors = []

    for fp in filepaths:
        try:
            source = Path(fp).read_text(encoding="utf-8", errors="replace")
            blocks = extract_blocks(source, fp, min_lines)
            all_blocks.extend(blocks)
        except (OSError, UnicodeDecodeError) as e:
            errors.append({"file": fp, "error": str(e)})

    clone_pairs = find_clones(all_blocks, threshold)
    clusters = cluster_clones(all_blocks, clone_pairs)
    score = compute_clone_score(len(all_blocks), clusters)

    return ScanResult(
        files_scanned=len(filepaths),
        blocks_extracted=len(all_blocks),
        clone_pairs=clone_pairs,
        errors=errors,
        clusters=clusters,
        clone_score=score,
    )


def scan_directory(
    path: str,
    min_lines: int = DEFAULT_MIN_LINES,
    threshold: float = DEFAULT_THRESHOLD,
    max_files: int = DEFAULT_MAX_FILES,
) -> ScanResult:
    """Scan a directory for code clones."""
    files = collect_python_files(path, max_files)
    return scan_files(files, min_lines, threshold)


# --- Output formatting ---

def _relative_path(filepath: str, base: str) -> str:
    """Make filepath relative to base for cleaner output."""
    try:
        return os.path.relpath(filepath, base)
    except ValueError:
        return filepath


def format_text(result: ScanResult, base_path: str = ".") -> str:
    """Format scan results as human-readable text."""
    lines = []
    grade = score_to_grade(result.clone_score)
    lines.append(f"Clone Detection Report")
    lines.append(f"=" * 50)
    lines.append(f"Score: {result.clone_score}/100 ({grade})")
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Blocks extracted: {result.blocks_extracted}")
    lines.append(f"Clone pairs found: {len(result.clone_pairs)}")
    clusters = result.clusters or []
    lines.append(f"Clone clusters: {len(clusters)}")
    total_dup = sum(c.total_duplicated_lines for c in clusters)
    if total_dup > 0:
        lines.append(f"Duplicated lines: {total_dup}")
    lines.append("")

    if not result.clone_pairs and not result.errors:
        lines.append("No clones detected. Clean codebase!")
        return "\n".join(lines)

    if not result.clone_pairs:
        lines.append("No clones detected.")
        lines.append("")

    # Show clusters (grouped view)
    if clusters:
        lines.append("## Clone Clusters (refactoring targets)")
        lines.append("")
        for cluster in clusters:
            dup = cluster.total_duplicated_lines
            lines.append(f"  Cluster #{cluster.cluster_id} — {cluster.size} blocks, "
                         f"{cluster.dominant_type}, ~{dup} duplicated lines")
            for block in cluster.blocks:
                rel = _relative_path(block.filepath, base_path)
                lines.append(f"    {block.name} ({rel}:{block.start_line}-{block.end_line}, "
                             f"{block.line_count} lines)")
            lines.append("")

    # Group by clone type (detail view)
    by_type = {"type-1": [], "type-2": [], "type-3": []}
    for cp in result.clone_pairs:
        by_type[cp.clone_type].append(cp)

    type_labels = {
        "type-1": "Type-1 (Exact copies)",
        "type-2": "Type-2 (Renamed identifiers)",
        "type-3": "Type-3 (Modified statements)",
    }

    for ctype in ["type-1", "type-2", "type-3"]:
        pairs = by_type[ctype]
        if not pairs:
            continue
        lines.append(f"## {type_labels[ctype]} — {len(pairs)} pair(s)")
        lines.append("")
        for cp in pairs:
            rel_a = _relative_path(cp.block_a.filepath, base_path)
            rel_b = _relative_path(cp.block_b.filepath, base_path)
            lines.append(f"  {cp.block_a.name} ({rel_a}:{cp.block_a.start_line}-{cp.block_a.end_line})")
            lines.append(f"  {cp.block_b.name} ({rel_b}:{cp.block_b.start_line}-{cp.block_b.end_line})")
            lines.append(f"  Similarity: {cp.similarity:.1%}  |  Lines: {cp.block_a.line_count} / {cp.block_b.line_count}")
            lines.append("")

    if result.errors:
        lines.append(f"## Errors ({len(result.errors)})")
        for err in result.errors:
            lines.append(f"  {err['file']}: {err['error']}")

    return "\n".join(lines)


def format_json(result: ScanResult, base_path: str = ".") -> str:
    """Format scan results as JSON."""
    clusters = result.clusters or []
    data = {
        "clone_score": result.clone_score,
        "grade": score_to_grade(result.clone_score),
        "files_scanned": result.files_scanned,
        "blocks_extracted": result.blocks_extracted,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "dominant_type": c.dominant_type,
                "max_similarity": round(c.max_similarity, 4),
                "min_similarity": round(c.min_similarity, 4),
                "duplicated_lines": c.total_duplicated_lines,
                "blocks": [
                    {
                        "name": b.name,
                        "filepath": _relative_path(b.filepath, base_path),
                        "start_line": b.start_line,
                        "end_line": b.end_line,
                        "line_count": b.line_count,
                        "block_type": b.block_type,
                    }
                    for b in c.blocks
                ],
            }
            for c in clusters
        ],
        "clone_pairs": [
            {
                "block_a": {
                    "name": cp.block_a.name,
                    "filepath": _relative_path(cp.block_a.filepath, base_path),
                    "start_line": cp.block_a.start_line,
                    "end_line": cp.block_a.end_line,
                    "line_count": cp.block_a.line_count,
                    "block_type": cp.block_a.block_type,
                },
                "block_b": {
                    "name": cp.block_b.name,
                    "filepath": _relative_path(cp.block_b.filepath, base_path),
                    "start_line": cp.block_b.start_line,
                    "end_line": cp.block_b.end_line,
                    "line_count": cp.block_b.line_count,
                    "block_type": cp.block_b.block_type,
                },
                "similarity": round(cp.similarity, 4),
                "clone_type": cp.clone_type,
            }
            for cp in result.clone_pairs
        ],
        "errors": result.errors,
    }
    return json.dumps(data, indent=2)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-clone-detect",
        description="AST-based Python code clone detector",
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Directory or file to scan (default: current directory)",
    )
    parser.add_argument(
        "-f", "--files", nargs="+", default=None,
        help="Specific files to scan",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold 0.0-1.0 (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-lines", type=int, default=DEFAULT_MIN_LINES,
        help=f"Minimum block size in lines (default: {DEFAULT_MIN_LINES})",
    )
    parser.add_argument(
        "--max-files", type=int, default=DEFAULT_MAX_FILES,
        help=f"Maximum files to scan (default: {DEFAULT_MAX_FILES})",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.files:
        result = scan_files(args.files, args.min_lines, args.threshold)
        base = "."
    else:
        target = args.path
        if not os.path.exists(target):
            print(f"Error: path not found: {target}", file=sys.stderr)
            return 1
        result = scan_directory(target, args.min_lines, args.threshold, args.max_files)
        base = target

    if args.json_output:
        print(format_json(result, base))
    else:
        print(format_text(result, base))

    return 0 if not result.clone_pairs else len(result.clone_pairs)


if __name__ == "__main__":
    sys.exit(main())
