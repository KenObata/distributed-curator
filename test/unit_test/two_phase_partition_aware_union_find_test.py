"""
Integration tests for two-phase partition-aware Union-Find.

Tests Phase 1 (partition-local UF via mapPartitions) and Phase 2 (cross-partition merge via SQL).
Requires SparkSession — these are integration tests, not pure unit tests.

Run: pytest test/integration_test/two_phase_union_find_test.py -v

Compare with test/unit_test/partition_local_union_find_test.py which tests
the UnionFind class in isolation (no Spark needed).
"""

import pytest
from pyspark.sql import Row, SparkSession

from two_phase_partition_aware_union_find import (
    Phase2GlobalTransitivityClosureQuery,
    run_phase1_local_union_find,
    run_phase2_global_transitivity_closure,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("TwoPhaseUnionFindTest")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.jars", "lib/graphframes-0.8.3-spark3.5-s_2.12.jar")  # <- add this
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def phase2_global_transitivity_closure_query(spark):
    """Fresh Phase2GlobalTransitivityClosureQuery instance per test."""
    return Phase2GlobalTransitivityClosureQuery(spark)


# =============================================================================
# Phase 1: Partition-local Union-Find
# =============================================================================


class TestPhase1:
    def test_output_schema(self, spark):
        """Output schema should be (doc_id: string, local_representative: string)."""
        pairs = spark.createDataFrame([Row(doc1="A", doc2="B", similarity=0.9, partition_id=0)]).repartition(1)

        result = run_phase1_local_union_find(pairs)
        assert result.columns == ["doc_id", "local_representative"]
        assert str(result.schema["doc_id"].dataType) == "StringType()"
        assert str(result.schema["local_representative"].dataType) == "StringType()"

    def test_output_size(self, spark):
        """
        Output length should be the total unique doc of input iterator.
        It's not equal to the size of iterator.

        Input:  (docA, docB) in partition 0
        Expect: 2 rows because (doc_id=docA, local_representative = docA), (doc_id=docB, local_representative=docA)
        """
        pairs = spark.createDataFrame([Row(doc1="docA", doc2="docB", similarity=0.9, partition_id=0)]).repartition(1)

        result = run_phase1_local_union_find(pairs).collect()
        assert len(result) == 2

    def test_single_pair_should_point_same_local_representative(self, spark):
        """
        One pair in one partition => both docs should map to same representative.

        Input:  (docA, docB) in partition 0
        Expect: docA => docA, docB => docA  (because UnionFind picks first arg)
        """
        pairs = spark.createDataFrame([Row(doc1="docA", doc2="docB", similarity=0.9, partition_id=0)]).repartition(1)
        result = run_phase1_local_union_find(pairs).collect()
        assert isinstance(result[0], Row)

        # Both docs should have the same local_representative
        assert result[0]["local_representative"] == result[1]["local_representative"]

    def test_multiple_and_transitive_pairs_should_point_same_local_representative(self, spark):
        """
        (A,B), (B,C) in same partition => all three map to same representative.
        Because union(B, C) doesn't union B and C — it unions find(B) and find(C), which is union(A, C)

        Input:  (A,B), (B,C) in partition 0
        Expect: A, B, C => same representative
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="B", doc2="C", similarity=0.9, partition_id=0),
            ]
        ).repartition(1)

        result = run_phase1_local_union_find(pairs).collect()

        assert len(result) == 3
        assert (
            result[0]["local_representative"] == result[1]["local_representative"] == result[2]["local_representative"]
        )

    def test_two_disconnected_pairs_in_same_partition(self, spark):
        """
        Two independent clusters in one partition => two different representatives.

        Input:  (A,B), (C,D) in partition 0
        Expect: A,B => same rep; C,D => same rep; two reps are different
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="C", doc2="D", similarity=0.9, partition_id=0),
            ]
        ).repartition(1)

        result = run_phase1_local_union_find(pairs).collect()
        root = {row["doc_id"]: row["local_representative"] for row in result}

        assert root["A"] == root["B"]
        assert root["C"] == root["D"]
        assert root["A"] != root["C"]

    def test_two_partitions_independent_diff_partition(self, spark):
        """
        Different pairs in different partitions => each partition runs UF independently.

        Partition 0: (A,B)
        Partition 1: (C,D)

        A doc might get different reps in different partitions — that's fine,
        Phase 2 merges them.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="C", doc2="D", similarity=0.9, partition_id=1),
            ]
        ).repartition(2, "partition_id")

        result = run_phase1_local_union_find(pairs).collect()
        reps = {row["doc_id"]: row["local_representative"] for row in result}

        assert reps["A"] == reps["B"]
        assert reps["C"] == reps["D"]
        assert reps["A"] != reps["C"]

    def test_duplicate_pairs_same_partition(self, spark):
        """
        Same pair appears twice in same partition => UF handles gracefully.
        Should still produce one entry per unique doc.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
            ]
        ).repartition(1)

        result = run_phase1_local_union_find(pairs).collect()
        reps = {row["doc_id"]: row["local_representative"] for row in result}

        assert len(reps) != 4  # not A -> A, A -> A, B->A, B->A
        assert len(reps) == 2
        assert reps["A"] == reps["B"]

    def test_same_doc_pair_appeared_different_partitions_gets_multiple_rows(self, spark):
        """
        Doc A appears in pairs in partition 0 AND partition 1.
        If Spark places them in different partitions, A appears twice.
        If hash partitioning collocates them, A appears once.
        Both are correct — Phase 2 handles either case.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="A", doc2="C", similarity=0.9, partition_id=1),
            ]
        ).repartition(2, "partition_id")

        result = run_phase1_local_union_find(pairs).collect()

        a_rows = [row for row in result if row["doc_id"] == "A"]
        assert len(a_rows) >= 1, "A should appear at least once"

        # Regardless of partition placement, B and C must appear
        doc_ids = {row["doc_id"] for row in result}
        assert {"A", "B", "C"} == doc_ids


# =============================================================================
# Phase 2: Cross-partition merge via SQL
# =============================================================================


class TestMultipleRepsEdgesQuery:
    def test_schema(self, spark, phase2_global_transitivity_closure_query):
        """Output should be (src: string, dst: string)."""
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
            ]
        )
        result, _ = phase2_global_transitivity_closure_query.multiple_reps_edges_query(local_results)
        assert result.columns == ["src", "dst"]
        result.unpersist()

    def test_single_rep_produces_no_edges(self, spark, phase2_global_transitivity_closure_query):
        """Doc with only one rep across partitions => no meta-edge."""
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="B", local_representative="X"),
            ]
        )
        result, count = phase2_global_transitivity_closure_query.multiple_reps_edges_query(local_results)
        assert count == 0
        result.unpersist()

    def test_two_reps_produces_one_edge(self, spark, phase2_global_transitivity_closure_query):
        """
        docA has rep X and Y from different partitions. Output should be (X,Y).
        reps[0] is anchor, slice(reps, 2, 100) gives the rest.
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
            ]
        )
        result, count = phase2_global_transitivity_closure_query.multiple_reps_edges_query(local_results)
        assert count == 1
        edge = result.collect()[0]
        assert {edge["src"], edge["dst"]} == {"X", "Y"}
        result.unpersist()

    def test_three_reps_produces_two_edges(self, spark, phase2_global_transitivity_closure_query):
        """
        docA has rep X, Y, Z => two meta-edges: (anchor, Y), (anchor, Z).
        Anchor is reps[0], rest are exploded via slice.
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
                Row(doc_id="A", local_representative="Z"),
            ]
        )
        result, count = phase2_global_transitivity_closure_query.multiple_reps_edges_query(local_results)
        assert count == 2
        result.unpersist()

    def test_multiple_docs_with_multiple_reps(self, spark, phase2_global_transitivity_closure_query):
        """
        docA => [X, Y], docB => [Y, Z].
        Two docs, each contributing one edge => 2 meta-edges.
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
                Row(doc_id="B", local_representative="Y"),
                Row(doc_id="B", local_representative="Z"),
            ]
        )
        result, count = phase2_global_transitivity_closure_query.multiple_reps_edges_query(local_results)
        assert count == 2
        result.unpersist()


class TestInitializeRepComponents:
    def test_each_rep_maps_to_itself(self, spark, phase2_global_transitivity_closure_query):
        """Each unique local_representative should map to itself as component."""
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="B", local_representative="X"),
                Row(doc_id="C", local_representative="Y"),
            ]
        )
        result = phase2_global_transitivity_closure_query.initialize_local_representative_component_columns(
            local_results
        )
        rows = {row["local_representative"]: row["component"] for row in result.collect()}

        assert len(rows) == 2  # (X,X) and (Y,Y)
        assert rows["X"] == "X"
        assert rows["Y"] == "Y"

    def test_schema(self, spark, phase2_global_transitivity_closure_query):
        """Output should be (local_representative, component)."""
        local_results = spark.createDataFrame([Row(doc_id="A", local_representative="X")])
        result = phase2_global_transitivity_closure_query.initialize_local_representative_component_columns(
            local_results
        )
        assert result.columns == ["local_representative", "component"]


class TestPropagateOneIteration:
    def test_simple_pair_converges_in_one_iteration(self, spark, phase2_global_transitivity_closure_query):
        """
        Single edge (X, Y). After one iteration, Y gets X.

        initial: (X, X), (Y, Y)
        multiple_reps_edges: (X, Y)

        2nd UNION ALL: multi.src=X matches init(X,X) => emits (Y, X)
        3rd UNION ALL: multi.dst=Y matches init(Y,Y) => emits (X, Y)

        Result: X=min(X,Y)=X, Y=min(Y,X)=X
        """
        initial_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
            ]
        )
        multiple_reps_edges = spark.createDataFrame([Row(src="X", dst="Y")])

        result = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components, multiple_reps_edges
        )
        rows = {row["local_representative"]: row["component"] for row in result.collect()}

        assert rows["X"] == "X"
        assert rows["Y"] == "X"

    def test_chain_iteration_1_partial_convergence(self, spark, phase2_global_transitivity_closure_query):
        """
        Chain X─Y─Z. After iteration 1, Z only reaches Y (not X).

        initial:  (X,X), (Y,Y), (Z,Z)
        multiple_reps_edges: (X,Y), (Y,Z)

        2nd UNION ALL:
          src=X matches init(X,X) => (Y, X)
          src=Y matches init(Y,Y) => (Z, Y)
        3rd UNION ALL:
          dst=Y matches init(Y,Y) => (X, Y)
          dst=Z matches init(Z,Z) => (Y, Z)

        GROUP BY MIN:
          X: min(X, Y)     = X
          Y: min(Y, X, Z)  = X
          Z: min(Z, Y)     = Y   <- NOT X yet!
        """
        initial_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
                Row(local_representative="Z", component="Z"),
            ]
        )
        multiple_reps_edges = spark.createDataFrame(
            [
                Row(src="X", dst="Y"),
                Row(src="Y", dst="Z"),
            ]
        )

        result_iter1 = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components, multiple_reps_edges
        )
        rows_iter1 = {row["local_representative"]: row["component"] for row in result_iter1.collect()}

        assert rows_iter1["X"] == "X"
        assert rows_iter1["Y"] == "X"
        assert rows_iter1["Z"] == "Y", "Z should only reach Y after iteration 1, not X"

    def test_chain_iteration_2_full_convergence(self, spark, phase2_global_transitivity_closure_query):
        """
        Chain X─Y─Z. Feed iteration 1 result into iteration 2.

        iteration 1 result: (X,X), (Y,X), (Z,Y)
        edges: (X,Y), (Y,Z)

        Iteration 2:
        2nd UNION ALL:
          src=X matches init(X,X) => (Y, X)
          src=Y matches init(Y,X) => (Z, X)   <- Z gets X via Y's updated component!
        3rd UNION ALL:
          dst=Y matches init(Y,X) => (X, X)
          dst=Z matches init(Z,Y) => (Y, Y)

        GROUP BY MIN:
          X: min(X, X)     = X
          Y: min(X, X, Y)  = X
          Z: min(Y, X)     = X   <- CONVERGED!
        """
        multiple_reps_edges = spark.createDataFrame(
            [
                Row(src="X", dst="Y"),
                Row(src="Y", dst="Z"),
            ]
        )

        # ── Iteration 1 ──
        initial_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
                Row(local_representative="Z", component="Z"),
            ]
        )
        result_iter1 = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components, multiple_reps_edges
        )
        rows_iter1 = {row["local_representative"]: row["component"] for row in result_iter1.collect()}
        # Verify iteration 1 state
        assert rows_iter1 == {"X": "X", "Y": "X", "Z": "Y"}

        # ── Iteration 2: feed result of iteration 1 as input ──
        result_iter2 = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            result_iter1, multiple_reps_edges
        )
        rows_iter2 = {row["local_representative"]: row["component"] for row in result_iter2.collect()}
        # All converged to X
        assert rows_iter2 == {"X": "X", "Y": "X", "Z": "X"}

    def test_disconnected_components_stay_separate(self, spark, phase2_global_transitivity_closure_query):
        """
        Two independent edges: (X,Y) and (W,V). They should never merge.
        """
        initial_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
                Row(local_representative="W", component="W"),
                Row(local_representative="V", component="V"),
            ]
        )
        multiple_reps_edges = spark.createDataFrame(
            [
                Row(src="X", dst="Y"),
                Row(src="W", dst="V"),
            ]
        )

        result = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components, multiple_reps_edges
        )
        rows = {row["local_representative"]: row["component"] for row in result.collect()}

        # Cluster 1
        assert rows["X"] == rows["Y"]
        # Cluster 2
        assert rows["V"] == rows["W"]
        # Separate
        assert rows["X"] != rows["V"]

    def test_no_multiple_reps_edges_no_change(self, spark, phase2_global_transitivity_closure_query):
        """No multiple_reps_edges => components unchanged."""
        initial_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
            ]
        )
        multiple_reps_edges = spark.createDataFrame([], "src string, dst string")

        result = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components, multiple_reps_edges
        )
        rows = {row["local_representative"]: row["component"] for row in result.collect()}

        assert rows["X"] == "X"
        assert rows["Y"] == "Y"

    def test_bidirectional_propagation(self, spark, phase2_global_transitivity_closure_query):
        """
        Edge stored as (X, Y). Both X and Y should receive each other's component.
        Tests that the 3rd UNION ALL (dst→src) works.

        Without 3rd UNION ALL: only Y would learn about X, not vice versa.
        """
        # Start with Y having a better (smaller) component than X
        initial_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="A"),  # A < X
            ]
        )
        multiple_reps_edges = spark.createDataFrame([Row(src="X", dst="Y")])

        result = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components, multiple_reps_edges
        )
        rows = {row["local_representative"]: row["component"] for row in result.collect()}

        # X should learn about A via the 3rd UNION ALL (dst=Y => src=X)
        assert rows["X"] == "A", "X should receive Y's component A via reverse propagation"
        assert rows["Y"] == "A"


class TestCountChanges:
    def test_output_schema(self, spark, phase2_global_transitivity_closure_query):
        """Return type is (hash_sum, distinct_components)."""
        df = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
                Row(local_representative="Z", component="Y"),
            ]
        )
        result = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        _, distinct_components = result
        assert distinct_components == 2  # X and Y

    def test_checksum_changes_when_component_changes(self, spark, phase2_global_transitivity_closure_query):
        """When components change between iterations, checksum must differ."""
        old = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="Y"),
            ]
        )
        new = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="X"),  # changed
            ]
        )
        old_hash, old_distinct = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(old)
        new_hash, new_distinct = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(new)
        assert old_hash != new_hash or old_distinct != new_distinct

    def test_checksum_unchanged_when_converged(self, spark, phase2_global_transitivity_closure_query):
        """When no components change, checksum must be identical."""
        df = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="X"),
            ]
        )
        hash1, distinct1 = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(df)
        hash2, distinct2 = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(df)
        assert hash1 == hash2
        assert distinct1 == distinct2

    def test_distinct_count_alone_insufficient(self, spark, phase2_global_transitivity_closure_query):
        """Distinct count can stay the same even when components change.
        hash_sum must catch this case."""
        before = spark.createDataFrame(
            [
                Row(local_representative="V", component="A"),
                Row(local_representative="W", component="B"),
                Row(local_representative="X", component="B"),
                Row(local_representative="Y", component="C"),
            ]
        )
        after = spark.createDataFrame(
            [
                Row(local_representative="V", component="A"),
                Row(local_representative="W", component="B"),
                Row(local_representative="X", component="A"),  # changed B -> A
                Row(local_representative="Y", component="C"),
            ]
        )
        _, before_distinct = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(before)
        _, after_distinct = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(after)
        # Distinct count is same (3 in both) — this is the false convergence case
        assert before_distinct == after_distinct == 3
        # But hash_sum differs
        before_hash, _ = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(before)
        after_hash, _ = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(after)
        assert before_hash != after_hash


class TestFinalMapping:
    def test_singletons_get_self_representative(self, spark, phase2_global_transitivity_closure_query):
        """Docs not in local_results => representative_id is doc_id."""
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="A"),
                Row(doc_id="B", local_representative="A"),
            ]
        )
        rep_components = spark.createDataFrame([Row(local_representative="A", component="A")])
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        result = phase2_global_transitivity_closure_query.map_final_doc_idto_global_representative(
            vertices, local_results, rep_components
        )
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["A"] == "A"
        assert reps["B"] == "A"
        assert reps["C"] == "C"

    def test_output_schema(self, spark, phase2_global_transitivity_closure_query):
        local_results = spark.createDataFrame([Row(doc_id="A", local_representative="A")])
        rep_components = spark.createDataFrame([Row(local_representative="A", component="A")])
        vertices = spark.createDataFrame([Row(id="A")])

        result = phase2_global_transitivity_closure_query.map_final_doc_idto_global_representative(
            vertices, local_results, rep_components
        )
        assert result.columns == ["doc_id", "representative_id"]

    def test_output_has_all_vertices(self, spark, phase2_global_transitivity_closure_query):
        """vertices (= input_df) is the main table
        so output must contain every vertex, not just those with pairs."""
        local_results = spark.createDataFrame([Row(doc_id="A", local_representative="A")])
        rep_components = spark.createDataFrame([Row(local_representative="A", component="A")])
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D")])

        result = phase2_global_transitivity_closure_query.map_final_doc_idto_global_representative(
            vertices, local_results, rep_components
        )
        result_ids = {row["doc_id"] for row in result.collect()}
        assert result_ids == {"A", "B", "C", "D"}

    def test_merged_components_reflected_in_output(self, spark, phase2_global_transitivity_closure_query):
        """
        After Phase 2 merging, rep Y got component X.
        Docs that had local_rep=Y should now get representative_id=X.
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="B", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
                Row(doc_id="C", local_representative="Y"),
            ]
        )
        # After Phase 2 convergence: Y merged into X
        rep_components = spark.createDataFrame(
            [
                Row(local_representative="X", component="X"),
                Row(local_representative="Y", component="X"),
            ]
        )
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        result = phase2_global_transitivity_closure_query.map_final_doc_idto_global_representative(
            vertices, local_results, rep_components
        )
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["A"] == "X"
        assert reps["B"] == "X"
        assert reps["C"] == "X"


class TestEndToEnd:
    def test_single_cluster_single_partition(self, spark):
        """
        A~B~C all in the same partition. Phase 1 resolves everything.
        Phase 2 is a no-op. D is singleton.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="B", doc2="C", similarity=0.85, partition_id=0),
            ]
        ).repartition(1)
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D")])

        local_results = run_phase1_local_union_find(pairs)
        result = {row["doc_id"]: row["local_representative"] for row in local_results.collect()}
        assert result["A"] == result["B"] == result["C"]

        result = run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["A"] == reps["B"] == reps["C"]
        assert reps["D"] == "D"

    def test_cross_partition_merge(self, spark):
        """
        Doc A in partition 0 with B, and partition 1 with C.
        Phase 1 can't connect B and C. Phase 2 merges them via A.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="A", doc2="C", similarity=0.85, partition_id=1),
            ]
        ).repartition(2, "partition_id")
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        local_results = run_phase1_local_union_find(pairs)
        result = run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=10)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["A"] == reps["B"] == reps["C"]

    def test_two_independent_clusters(self, spark):
        """Two clusters that should never merge. E is singleton."""
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="C", doc2="D", similarity=0.9, partition_id=1),
            ]
        ).repartition(2, "partition_id")
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D"), Row(id="E")])

        local_results = run_phase1_local_union_find(pairs)
        result = run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["A"] == reps["B"]
        assert reps["C"] == reps["D"]
        assert reps["A"] != reps["C"]
        assert reps["E"] == "E"
