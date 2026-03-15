"""Tests for the clone detection engine."""
from __future__ import annotations

import ast
import textwrap

import pytest

from code_health_suite.engines.clone_detect import (
    CodeBlock,
    CloneCluster,
    ClonePair,
    ScanResult,
    _build_ngram_set,
    _is_nested,
    _jaccard_similarity,
    _Normalizer,
    classify_clone,
    cluster_clones,
    collect_python_files,
    compute_clone_score,
    compute_similarity,
    extract_blocks,
    find_clones,
    format_json,
    format_text,
    normalize_ast,
    normalize_source,
    scan_files,
    score_to_grade,
    _count_nodes,
    build_parser,
    main,
    DEFAULT_THRESHOLD,
    DEFAULT_MIN_LINES,
)


# --- Helpers ---

def _make_block(
    name: str = "func_a",
    filepath: str = "a.py",
    start_line: int = 1,
    end_line: int = 10,
    source: str = "def func_a(): pass",
    normalized: str = "def V0(): pass",
    block_type: str = "function",
    node_count: int = 5,
) -> CodeBlock:
    """Create a CodeBlock for testing."""
    return CodeBlock(
        name=name,
        filepath=filepath,
        start_line=start_line,
        end_line=end_line,
        source=source,
        normalized=normalized,
        block_type=block_type,
        node_count=node_count,
    )


SIMPLE_FUNC = textwrap.dedent("""\
    def add(a, b):
        result = a + b
        if result > 0:
            return result
        else:
            return -result
""")

CLONE_FUNC = textwrap.dedent("""\
    def subtract(x, y):
        result = x - y
        if result > 0:
            return result
        else:
            return -result
""")

DIFFERENT_FUNC = textwrap.dedent("""\
    def multiply(a, b, c):
        for i in range(a):
            for j in range(b):
                c += i * j
                if c > 100:
                    break
        return c
""")

TWO_FUNC_MODULE = textwrap.dedent("""\
    def func_one(a, b):
        result = a + b
        if result > 0:
            return result
        else:
            return -result

    def func_two(x, y):
        result = x + y
        if result > 0:
            return result
        else:
            return -result
""")


# =============================================================================
# CodeBlock dataclass
# =============================================================================

class TestCodeBlock:
    def test_line_count(self):
        block = _make_block(start_line=5, end_line=15)
        assert block.line_count == 11

    def test_line_count_single_line(self):
        block = _make_block(start_line=1, end_line=1)
        assert block.line_count == 1


# =============================================================================
# CloneCluster properties
# =============================================================================

class TestCloneCluster:
    def test_size(self):
        blocks = [_make_block(name="a"), _make_block(name="b"), _make_block(name="c")]
        cluster = CloneCluster(
            cluster_id=0, blocks=blocks,
            max_similarity=1.0, min_similarity=0.9, dominant_type="type-1",
        )
        assert cluster.size == 3

    def test_total_duplicated_lines_empty(self):
        cluster = CloneCluster(
            cluster_id=0, blocks=[_make_block()],
            max_similarity=1.0, min_similarity=1.0, dominant_type="type-1",
        )
        assert cluster.total_duplicated_lines == 0

    def test_total_duplicated_lines(self):
        b1 = _make_block(start_line=1, end_line=10)  # 10 lines
        b2 = _make_block(start_line=1, end_line=8)   # 8 lines
        b3 = _make_block(start_line=1, end_line=6)   # 6 lines
        cluster = CloneCluster(
            cluster_id=0, blocks=[b1, b2, b3],
            max_similarity=1.0, min_similarity=0.9, dominant_type="type-2",
        )
        # Keep longest (10), others are duplicated: 8 + 6 = 14
        assert cluster.total_duplicated_lines == 14


# =============================================================================
# AST normalization
# =============================================================================

class TestNormalization:
    def test_count_nodes(self):
        tree = ast.parse("x = 1 + 2")
        count = _count_nodes(tree)
        assert count > 1  # Module + Assign + ...

    def test_normalizer_renames_variables(self):
        normalizer = _Normalizer()
        tree = ast.parse("x = 1")
        normalizer.visit(tree)
        # After normalization, x should become V0
        dumped = ast.dump(tree)
        assert "V0" in dumped

    def test_normalizer_consistent_mapping(self):
        normalizer = _Normalizer()
        tree = ast.parse("x = x + y")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        # x → V0, y → V1 (consistent mapping)
        assert "V0" in dumped
        assert "V1" in dumped

    def test_normalize_ast_identical_structure(self):
        """Two functions with same structure but different names should normalize the same."""
        code_a = "def foo(a): return a + 1"
        code_b = "def bar(x): return x + 1"
        tree_a = ast.parse(code_a).body[0]
        tree_b = ast.parse(code_b).body[0]
        assert normalize_ast(tree_a) == normalize_ast(tree_b)

    def test_normalize_ast_different_structure(self):
        """Different structure should produce different normalized forms."""
        code_a = "def foo(a): return a + 1"
        code_b = "def bar(a, b): return a * b"
        tree_a = ast.parse(code_a).body[0]
        tree_b = ast.parse(code_b).body[0]
        assert normalize_ast(tree_a) != normalize_ast(tree_b)

    def test_normalize_source_returns_string(self):
        tree = ast.parse("def f(x): return x").body[0]
        result = normalize_source(tree)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalizer_handles_constants(self):
        normalizer = _Normalizer()
        tree = ast.parse("x = 'hello'\ny = 42")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        # String constants → "S", numbers → 0
        assert "'S'" in dumped or "value='S'" in dumped

    def test_normalizer_handles_async(self):
        normalizer = _Normalizer()
        tree = ast.parse("async def foo(): pass")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        assert "V0" in dumped

    def test_normalizer_handles_attribute(self):
        normalizer = _Normalizer()
        tree = ast.parse("obj.method()")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        # obj → V0, method → V1
        assert "V0" in dumped

    def test_normalizer_handles_keyword(self):
        normalizer = _Normalizer()
        tree = ast.parse("f(key=1)")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        assert "V" in dumped

    def test_normalizer_handles_alias(self):
        normalizer = _Normalizer()
        tree = ast.parse("import os as operating_system")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        assert "V0" in dumped
        assert "V1" in dumped

    def test_normalizer_class_def(self):
        normalizer = _Normalizer()
        tree = ast.parse("class Foo: pass")
        normalizer.visit(tree)
        dumped = ast.dump(tree)
        assert "V0" in dumped


# =============================================================================
# Block extraction
# =============================================================================

class TestExtractBlocks:
    def test_extract_simple_function(self):
        blocks = extract_blocks(SIMPLE_FUNC, "test.py", min_lines=3)
        assert len(blocks) == 1
        assert blocks[0].name == "add"
        assert blocks[0].block_type == "function"
        assert blocks[0].filepath == "test.py"

    def test_extract_respects_min_lines(self):
        short_func = "def f(): return 1"
        blocks = extract_blocks(short_func, "test.py", min_lines=5)
        assert len(blocks) == 0

    def test_extract_multiple_functions(self):
        blocks = extract_blocks(TWO_FUNC_MODULE, "test.py", min_lines=3)
        assert len(blocks) == 2
        names = {b.name for b in blocks}
        assert names == {"func_one", "func_two"}

    def test_extract_class_with_body(self):
        code = textwrap.dedent("""\
            class MyClass:
                x = 1
                y = 2
                z = 3
                def method(self):
                    result = self.x + self.y
                    if result > 0:
                        return result
                    return -result
                def other(self):
                    result = self.y + self.z
                    if result > 0:
                        return result
                    return -result
        """)
        blocks = extract_blocks(code, "test.py", min_lines=3)
        # Should extract class + methods that meet min_lines
        names = {b.name for b in blocks}
        assert "MyClass" in names
        assert "method" in names or "other" in names

    def test_extract_skips_thin_class(self):
        """Class with <2 non-method statements should be skipped."""
        code = textwrap.dedent("""\
            class Thin:
                def method_a(self):
                    return 1
                def method_b(self):
                    return 2
        """)
        blocks = extract_blocks(code, "test.py", min_lines=3)
        names = {b.name for b in blocks}
        assert "Thin" not in names

    def test_extract_syntax_error_returns_empty(self):
        blocks = extract_blocks("def broken(", "bad.py")
        assert blocks == []

    def test_extract_async_function(self):
        code = textwrap.dedent("""\
            async def fetch_data(url):
                response = await get(url)
                data = response.json()
                if data:
                    return data
                return None
        """)
        blocks = extract_blocks(code, "test.py", min_lines=3)
        assert len(blocks) == 1
        assert blocks[0].name == "fetch_data"

    def test_block_has_node_count(self):
        blocks = extract_blocks(SIMPLE_FUNC, "test.py", min_lines=3)
        assert blocks[0].node_count > 0

    def test_block_has_normalized(self):
        blocks = extract_blocks(SIMPLE_FUNC, "test.py", min_lines=3)
        assert len(blocks[0].normalized) > 0


# =============================================================================
# N-gram and Jaccard
# =============================================================================

class TestNgramJaccard:
    def test_build_ngram_set_normal(self):
        ngrams = _build_ngram_set("abcdef", 4)
        assert isinstance(ngrams, frozenset)
        assert len(ngrams) == 3  # "abcd", "bcde", "cdef"

    def test_build_ngram_set_short_string(self):
        ngrams = _build_ngram_set("ab", 4)
        assert ngrams == frozenset(["ab"])

    def test_build_ngram_set_empty(self):
        ngrams = _build_ngram_set("", 4)
        assert ngrams == frozenset()

    def test_jaccard_identical(self):
        s = frozenset(["a", "b", "c"])
        assert _jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint(self):
        a = frozenset(["a", "b"])
        b = frozenset(["c", "d"])
        assert _jaccard_similarity(a, b) == 0.0

    def test_jaccard_partial(self):
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        # intersection=2, union=4
        assert _jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_jaccard_empty(self):
        assert _jaccard_similarity(frozenset(), frozenset(["a"])) == 0.0
        assert _jaccard_similarity(frozenset(), frozenset()) == 0.0


# =============================================================================
# Similarity computation
# =============================================================================

class TestComputeSimilarity:
    def test_identical_blocks(self):
        block = _make_block(node_count=10, normalized="def V0(): return V1")
        sim = compute_similarity(block, block)
        assert sim == 1.0

    def test_zero_node_count(self):
        a = _make_block(node_count=0)
        b = _make_block(node_count=5)
        assert compute_similarity(a, b) == 0.0

    def test_very_different_node_counts_rejected(self):
        a = _make_block(node_count=100, normalized="x" * 100)
        b = _make_block(node_count=10, normalized="y" * 10)
        # Ratio 10/100 = 0.1, below threshold*0.7 = 0.56 → rejected
        assert compute_similarity(a, b) == 0.0

    def test_empty_normalized_rejected(self):
        a = _make_block(node_count=5, normalized="")
        b = _make_block(node_count=5, normalized="something")
        assert compute_similarity(a, b) == 0.0

    def test_similar_blocks_from_source(self):
        """Extract real blocks from similar functions and check similarity."""
        blocks_a = extract_blocks(SIMPLE_FUNC, "a.py", min_lines=3)
        blocks_b = extract_blocks(CLONE_FUNC, "b.py", min_lines=3)
        assert len(blocks_a) == 1 and len(blocks_b) == 1
        sim = compute_similarity(blocks_a[0], blocks_b[0])
        assert sim > 0.7  # These are structurally very similar

    def test_different_blocks_low_similarity(self):
        blocks_a = extract_blocks(SIMPLE_FUNC, "a.py", min_lines=3)
        blocks_b = extract_blocks(DIFFERENT_FUNC, "b.py", min_lines=3)
        assert len(blocks_a) == 1 and len(blocks_b) == 1
        sim = compute_similarity(blocks_a[0], blocks_b[0])
        assert sim < 0.8  # Structurally different

    def test_precomputed_ngrams(self):
        block = _make_block(node_count=10, normalized="def V0(): return V1 + V2")
        ngrams = _build_ngram_set(block.normalized)
        sim = compute_similarity(block, block, ngram_a=ngrams, ngram_b=ngrams)
        assert sim == 1.0


# =============================================================================
# Clone classification
# =============================================================================

class TestClassifyClone:
    def test_type_1_exact_copy(self):
        a = _make_block(source="def f(): return 1", normalized="N1")
        b = _make_block(source="def f(): return 1", normalized="N1")
        assert classify_clone(a, b, 1.0) == "type-1"

    def test_type_2_renamed(self):
        a = _make_block(source="def foo(): return x", normalized="N1")
        b = _make_block(source="def bar(): return y", normalized="N1")
        assert classify_clone(a, b, 1.0) == "type-2"

    def test_type_3_modified(self):
        a = _make_block(normalized="N1")
        b = _make_block(normalized="N2")
        assert classify_clone(a, b, 0.85) == "type-3"


# =============================================================================
# Nested detection
# =============================================================================

class TestIsNested:
    def test_different_files_not_nested(self):
        a = _make_block(filepath="a.py", start_line=1, end_line=10)
        b = _make_block(filepath="b.py", start_line=1, end_line=10)
        assert _is_nested(a, b) is False

    def test_same_start_line(self):
        a = _make_block(filepath="a.py", start_line=5, end_line=10)
        b = _make_block(filepath="a.py", start_line=5, end_line=8)
        assert _is_nested(a, b) is True

    def test_a_contains_b(self):
        a = _make_block(filepath="a.py", start_line=1, end_line=20)
        b = _make_block(filepath="a.py", start_line=5, end_line=10)
        assert _is_nested(a, b) is True

    def test_b_contains_a(self):
        a = _make_block(filepath="a.py", start_line=5, end_line=10)
        b = _make_block(filepath="a.py", start_line=1, end_line=20)
        assert _is_nested(a, b) is True

    def test_non_overlapping_same_file(self):
        a = _make_block(filepath="a.py", start_line=1, end_line=10)
        b = _make_block(filepath="a.py", start_line=15, end_line=25)
        assert _is_nested(a, b) is False


# =============================================================================
# Clone finding (integration)
# =============================================================================

class TestFindClones:
    def test_no_clones_in_different_code(self):
        blocks = extract_blocks(SIMPLE_FUNC, "a.py", min_lines=3)
        blocks += extract_blocks(DIFFERENT_FUNC, "b.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.9)
        assert len(clones) == 0

    def test_finds_clones_in_similar_code(self):
        blocks = extract_blocks(TWO_FUNC_MODULE, "test.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.7)
        assert len(clones) >= 1
        assert clones[0].similarity >= 0.7

    def test_empty_blocks(self):
        clones = find_clones([], threshold=0.8)
        assert clones == []

    def test_single_block_no_clones(self):
        blocks = extract_blocks(SIMPLE_FUNC, "a.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.8)
        assert clones == []

    def test_exact_copies_detected(self):
        """Same function in two files should be type-1."""
        blocks = extract_blocks(SIMPLE_FUNC, "a.py", min_lines=3)
        blocks += extract_blocks(SIMPLE_FUNC, "b.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.8)
        assert len(clones) >= 1
        # Exact same code → type-1
        assert any(c.clone_type == "type-1" for c in clones)

    def test_renamed_copies_detected(self):
        """Same structure, different names → detected as clones."""
        blocks = extract_blocks(SIMPLE_FUNC, "a.py", min_lines=3)
        blocks += extract_blocks(CLONE_FUNC, "b.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.7)
        assert len(clones) >= 1
        # The similarity should be high since the structure is nearly identical
        assert clones[0].similarity >= 0.7

    def test_clones_sorted_by_similarity(self):
        blocks = extract_blocks(TWO_FUNC_MODULE, "test.py", min_lines=3)
        blocks += extract_blocks(SIMPLE_FUNC, "other.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.7)
        if len(clones) > 1:
            for i in range(len(clones) - 1):
                assert clones[i].similarity >= clones[i + 1].similarity

    def test_nested_blocks_excluded(self):
        """A function inside a class should not clone-match with the class itself."""
        code = textwrap.dedent("""\
            class Container:
                x = 1
                y = 2
                z = 3
                def method(self):
                    result = self.x + self.y
                    if result > 0:
                        return result
                    return -result
        """)
        blocks = extract_blocks(code, "test.py", min_lines=3)
        clones = find_clones(blocks, threshold=0.5)
        # Should not report nested blocks as clones of each other
        for c in clones:
            assert not _is_nested(c.block_a, c.block_b)


# =============================================================================
# Clustering
# =============================================================================

class TestClustering:
    def test_empty_pairs(self):
        clusters = cluster_clones([], [])
        assert clusters == []

    def test_single_pair_creates_cluster(self):
        a = _make_block(name="a", start_line=1, end_line=10)
        b = _make_block(name="b", start_line=20, end_line=30)
        pair = ClonePair(block_a=a, block_b=b, similarity=0.9, clone_type="type-2")
        clusters = cluster_clones([a, b], [pair])
        assert len(clusters) == 1
        assert clusters[0].size == 2

    def test_transitive_clustering(self):
        """If A≈B and B≈C, they should all be in one cluster."""
        a = _make_block(name="a", filepath="1.py", start_line=1, end_line=10)
        b = _make_block(name="b", filepath="2.py", start_line=1, end_line=10)
        c = _make_block(name="c", filepath="3.py", start_line=1, end_line=10)
        pair_ab = ClonePair(block_a=a, block_b=b, similarity=0.9, clone_type="type-2")
        pair_bc = ClonePair(block_a=b, block_b=c, similarity=0.85, clone_type="type-3")
        clusters = cluster_clones([a, b, c], [pair_ab, pair_bc])
        assert len(clusters) == 1
        assert clusters[0].size == 3

    def test_separate_clusters(self):
        a = _make_block(name="a", filepath="1.py", start_line=1, end_line=10)
        b = _make_block(name="b", filepath="2.py", start_line=1, end_line=10)
        c = _make_block(name="c", filepath="3.py", start_line=1, end_line=10)
        d = _make_block(name="d", filepath="4.py", start_line=1, end_line=10)
        pair_ab = ClonePair(block_a=a, block_b=b, similarity=0.9, clone_type="type-1")
        pair_cd = ClonePair(block_a=c, block_b=d, similarity=0.85, clone_type="type-3")
        clusters = cluster_clones([a, b, c, d], [pair_ab, pair_cd])
        assert len(clusters) == 2

    def test_cluster_dominant_type(self):
        a = _make_block(name="a", filepath="1.py", start_line=1, end_line=10)
        b = _make_block(name="b", filepath="2.py", start_line=1, end_line=10)
        pair = ClonePair(block_a=a, block_b=b, similarity=1.0, clone_type="type-1")
        clusters = cluster_clones([a, b], [pair])
        assert clusters[0].dominant_type == "type-1"

    def test_clusters_sorted_by_duplicated_lines(self):
        # Cluster 1: small blocks (5 lines each)
        a = _make_block(name="a", filepath="1.py", start_line=1, end_line=5)
        b = _make_block(name="b", filepath="2.py", start_line=1, end_line=5)
        # Cluster 2: large blocks (50 lines each)
        c = _make_block(name="c", filepath="3.py", start_line=1, end_line=50)
        d = _make_block(name="d", filepath="4.py", start_line=1, end_line=50)
        pair_ab = ClonePair(block_a=a, block_b=b, similarity=0.9, clone_type="type-2")
        pair_cd = ClonePair(block_a=c, block_b=d, similarity=0.85, clone_type="type-3")
        clusters = cluster_clones([a, b, c, d], [pair_ab, pair_cd])
        assert len(clusters) == 2
        # Larger duplicated lines cluster first
        assert clusters[0].total_duplicated_lines >= clusters[1].total_duplicated_lines


# =============================================================================
# Clone score
# =============================================================================

class TestCloneScore:
    def test_perfect_score_no_clusters(self):
        assert compute_clone_score(10, []) == 100

    def test_perfect_score_zero_blocks(self):
        assert compute_clone_score(0, []) == 100

    def test_type_1_costs_more(self):
        """Type-1 clones should penalize more than type-3."""
        blocks_t1 = [_make_block(start_line=1, end_line=10), _make_block(start_line=1, end_line=10)]
        cluster_t1 = CloneCluster(
            cluster_id=0, blocks=blocks_t1,
            max_similarity=1.0, min_similarity=1.0, dominant_type="type-1",
        )
        cluster_t3 = CloneCluster(
            cluster_id=0, blocks=blocks_t1,
            max_similarity=0.85, min_similarity=0.85, dominant_type="type-3",
        )
        score_t1 = compute_clone_score(10, [cluster_t1])
        score_t3 = compute_clone_score(10, [cluster_t3])
        assert score_t1 < score_t3  # Type-1 more penalty

    def test_score_floors_at_zero(self):
        """Massive duplication should floor at 0, not go negative."""
        blocks = [_make_block(start_line=1, end_line=1000) for _ in range(10)]
        cluster = CloneCluster(
            cluster_id=0, blocks=blocks,
            max_similarity=1.0, min_similarity=1.0, dominant_type="type-1",
        )
        score = compute_clone_score(10, [cluster])
        assert score == 0


class TestScoreToGrade:
    @pytest.mark.parametrize("score,expected", [
        (100, "A"), (95, "A"), (90, "A"),
        (89, "B"), (85, "B"), (80, "B"),
        (79, "C"), (75, "C"), (70, "C"),
        (69, "D"), (65, "D"), (60, "D"),
        (59, "F"), (30, "F"), (0, "F"),
    ])
    def test_grade_boundaries(self, score, expected):
        assert score_to_grade(score) == expected


# =============================================================================
# File collection
# =============================================================================

class TestCollectPythonFiles:
    def test_collect_single_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        files = collect_python_files(str(f))
        assert len(files) == 1
        assert files[0] == str(f)

    def test_collect_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.txt").write_text("not python")
        files = collect_python_files(str(tmp_path))
        assert len(files) == 2

    def test_skips_pycache(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.py").write_text("cached = True")
        (tmp_path / "real.py").write_text("real = True")
        files = collect_python_files(str(tmp_path))
        assert len(files) == 1

    def test_respects_max_files(self, tmp_path):
        for i in range(20):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}")
        files = collect_python_files(str(tmp_path), max_files=5)
        assert len(files) == 5

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / "myenv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr")
        (venv / "script.py").write_text("venv_code = True")
        (tmp_path / "real.py").write_text("real = True")
        files = collect_python_files(str(tmp_path))
        assert len(files) == 1

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("s = 1")
        (tmp_path / "visible.py").write_text("v = 1")
        files = collect_python_files(str(tmp_path))
        assert len(files) == 1


# =============================================================================
# Scan integration
# =============================================================================

class TestScanFiles:
    def test_scan_cloned_files(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text(SIMPLE_FUNC)
        f2.write_text(SIMPLE_FUNC)
        result = scan_files([str(f1), str(f2)], min_lines=3, threshold=0.8)
        assert result.files_scanned == 2
        assert result.blocks_extracted >= 2
        assert len(result.clone_pairs) >= 1
        assert result.clusters is not None

    def test_scan_no_clones(self, tmp_path):
        f = tmp_path / "unique.py"
        f.write_text(SIMPLE_FUNC)
        result = scan_files([str(f)], min_lines=3, threshold=0.8)
        assert result.files_scanned == 1
        assert len(result.clone_pairs) == 0
        assert result.clone_score == 100

    def test_scan_handles_read_errors(self, tmp_path):
        result = scan_files(["/nonexistent/file.py"], min_lines=3, threshold=0.8)
        assert len(result.errors) == 1

    def test_scan_empty_files(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = scan_files([str(f)], min_lines=3)
        assert result.files_scanned == 1
        assert result.blocks_extracted == 0


# =============================================================================
# Output formatting
# =============================================================================

class TestFormatText:
    def test_clean_codebase(self):
        result = ScanResult(
            files_scanned=5, blocks_extracted=10,
            clone_pairs=[], errors=[], clusters=[], clone_score=100,
        )
        text = format_text(result)
        assert "Score: 100/100" in text
        assert "Clean codebase" in text

    def test_with_clones(self):
        a = _make_block(name="func_a", filepath="a.py")
        b = _make_block(name="func_b", filepath="b.py")
        pair = ClonePair(block_a=a, block_b=b, similarity=0.9, clone_type="type-2")
        cluster = CloneCluster(
            cluster_id=0, blocks=[a, b],
            max_similarity=0.9, min_similarity=0.9, dominant_type="type-2",
        )
        result = ScanResult(
            files_scanned=2, blocks_extracted=2,
            clone_pairs=[pair], errors=[], clusters=[cluster], clone_score=85,
        )
        text = format_text(result)
        assert "Score: 85/100" in text
        assert "type-2" in text.lower() or "Type-2" in text

    def test_with_errors(self):
        result = ScanResult(
            files_scanned=1, blocks_extracted=0,
            clone_pairs=[], errors=[{"file": "bad.py", "error": "read error"}],
            clusters=[], clone_score=100,
        )
        text = format_text(result)
        assert "bad.py" in text
        assert "Errors" in text


class TestFormatJson:
    def test_json_structure(self):
        result = ScanResult(
            files_scanned=1, blocks_extracted=0,
            clone_pairs=[], errors=[], clusters=[], clone_score=100,
        )
        import json
        data = json.loads(format_json(result))
        assert data["clone_score"] == 100
        assert data["grade"] == "A"
        assert data["clusters"] == []
        assert data["clone_pairs"] == []

    def test_json_with_clones(self):
        a = _make_block(name="fa", filepath="a.py")
        b = _make_block(name="fb", filepath="b.py")
        pair = ClonePair(block_a=a, block_b=b, similarity=0.95, clone_type="type-1")
        cluster = CloneCluster(
            cluster_id=0, blocks=[a, b],
            max_similarity=0.95, min_similarity=0.95, dominant_type="type-1",
        )
        result = ScanResult(
            files_scanned=2, blocks_extracted=2,
            clone_pairs=[pair], errors=[], clusters=[cluster], clone_score=97,
        )
        import json
        data = json.loads(format_json(result))
        assert len(data["clone_pairs"]) == 1
        assert data["clone_pairs"][0]["similarity"] == 0.95
        assert len(data["clusters"]) == 1


# =============================================================================
# CLI
# =============================================================================

class TestCLI:
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.path == "."
        assert args.threshold == DEFAULT_THRESHOLD
        assert args.min_lines == DEFAULT_MIN_LINES

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args(["src/", "--threshold", "0.9", "--min-lines", "10"])
        assert args.path == "src/"
        assert args.threshold == 0.9
        assert args.min_lines == 10

    def test_main_nonexistent_path(self):
        ret = main(["/nonexistent/path/here"])
        assert ret == 1

    def test_main_clean_directory(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("def unique_function():\n    return 42\n")
        ret = main([str(tmp_path)])
        assert ret == 0  # No clones

    def test_main_with_clones(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text(SIMPLE_FUNC)
        f2.write_text(SIMPLE_FUNC)
        ret = main([str(tmp_path), "--min-lines", "3"])
        assert ret >= 1  # Clone pairs found

    def test_main_json_output(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text(TWO_FUNC_MODULE)
        main([str(tmp_path), "--json", "--min-lines", "3"])
        captured = capsys.readouterr()
        import json
        data = json.loads(captured.out)
        assert "clone_score" in data

    def test_main_specific_files(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text(SIMPLE_FUNC)
        f2.write_text(CLONE_FUNC)
        ret = main(["-f", str(f1), str(f2), "--min-lines", "3"])
        assert isinstance(ret, int)
