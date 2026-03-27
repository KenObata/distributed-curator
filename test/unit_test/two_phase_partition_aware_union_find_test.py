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
    _run_phase1_local_union_find,
    _run_phase2_global_transitivity_closure,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("TwoPhaseUnionFindTest")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.jars", "lib/graphframes-0.8.3-spark3.5-s_2.12.jar")  # ← add this
        .getOrCreate()
    )
    yield session
    session.stop()


# =============================================================================
# Phase 1: Partition-local Union-Find
# =============================================================================


class TestPhase1:
    def test_single_pair_same_partition(self, spark):
        """
        One pair in one partition → both docs map to same representative.

        Input:  (docA, docB) in partition 0
        Expect: docA → docA, docB → docA  (or both → docB, depends on UF ordering)
        """
        pairs = spark.createDataFrame([Row(doc1="docA", doc2="docB", similarity=0.9, partition_id=0)]).repartition(1)

        result = _run_phase1_local_union_find(pairs).collect()

        # Both docs should have the same local_representative
        reps = {row["doc_id"]: row["local_representative"] for row in result}
        assert len(reps) == 2
        assert reps["docA"] == reps["docB"]

    def test_transitive_chain_single_partition(self, spark):
        """
        A~B, B~C in same partition → all three map to same representative.

        Input:  (A,B), (B,C) in partition 0
        Expect: A, B, C → same representative
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="B", doc2="C", similarity=0.85, partition_id=0),
            ]
        ).repartition(1)

        result = _run_phase1_local_union_find(pairs).collect()
        reps = {row["doc_id"]: row["local_representative"] for row in result}

        assert len(reps) == 3
        assert reps["A"] == reps["B"] == reps["C"]

    def test_two_clusters_same_partition(self, spark):
        """
        Two independent clusters in one partition → two different representatives.

        Input:  (A,B), (C,D) in partition 0
        Expect: A,B → same rep; C,D → same rep; two reps are different
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="C", doc2="D", similarity=0.9, partition_id=0),
            ]
        ).repartition(1)

        result = _run_phase1_local_union_find(pairs).collect()
        reps = {row["doc_id"]: row["local_representative"] for row in result}

        assert reps["A"] == reps["B"]
        assert reps["C"] == reps["D"]
        assert reps["A"] != reps["C"]

    def test_two_partitions_independent(self, spark):
        """
        Different pairs in different partitions → each partition runs UF independently.

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

        result = _run_phase1_local_union_find(pairs).collect()
        reps = {row["doc_id"]: row["local_representative"] for row in result}

        assert reps["A"] == reps["B"]
        assert reps["C"] == reps["D"]
        assert reps["A"] != reps["C"]

    def test_output_schema(self, spark):
        """Output schema should be (doc_id: string, local_representative: string)."""
        pairs = spark.createDataFrame([Row(doc1="A", doc2="B", similarity=0.9, partition_id=0)]).repartition(1)

        result = _run_phase1_local_union_find(pairs)
        assert result.columns == ["doc_id", "local_representative"]
        assert str(result.schema["doc_id"].dataType) == "StringType()"
        assert str(result.schema["local_representative"].dataType) == "StringType()"

    def test_duplicate_pairs_same_partition(self, spark):
        """
        Same pair appears twice in same partition → UF handles gracefully.
        Should still produce one entry per unique doc.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="A", doc2="B", similarity=0.85, partition_id=0),
            ]
        ).repartition(1)

        result = _run_phase1_local_union_find(pairs).collect()
        reps = {row["doc_id"]: row["local_representative"] for row in result}

        assert len(reps) == 2
        assert reps["A"] == reps["B"]

    def test_same_doc_different_partitions_gets_multiple_rows(self, spark):
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

        result = _run_phase1_local_union_find(pairs).collect()

        a_rows = [row for row in result if row["doc_id"] == "A"]
        assert len(a_rows) >= 1, "A should appear at least once"

        # Regardless of partition placement, B and C must appear
        doc_ids = {row["doc_id"] for row in result}
        assert {"A", "B", "C"} == doc_ids


# =============================================================================
# Phase 2: Cross-partition merge via SQL
# =============================================================================


class TestPhase2:
    def test_no_cross_partition_edges(self, spark):
        """
        All docs have same local_rep across partitions → zero meta-edges.
        Should resolve locally without iteration.

        local_results:
          (A, rep_X), (B, rep_X)   ← both already agree
        vertices: A, B, C (C is singleton)
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="A"),
                Row(doc_id="B", local_representative="A"),
            ]
        )
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["A"] == reps["B"]  # same component
        assert reps["C"] == "C"  # singleton → self-representative

    def test_simple_cross_partition_merge(self, spark):
        """
        Doc A got rep_X in partition 0, rep_Y in partition 1.
        Phase 2 should merge X and Y into one component.

        local_results:
          (A, X), (B, X)     ← from partition 0
          (A, Y), (C, Y)     ← from partition 1

        After merge: A, B, C all map to min(X, Y).
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="B", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
                Row(doc_id="C", local_representative="Y"),
            ]
        )
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        # All three should have same representative
        assert reps["A"] == reps["B"] == reps["C"]
        # Representative should be min("X", "Y") = "X"
        assert reps["A"] == "X"

    def test_transitive_chain_across_partitions(self, spark):
        """
        X─Y from docA, Y─Z from docB → X, Y, Z must all merge.
        This requires 2+ iterations to converge.

        local_results:
          (A, X), (A, Y)     ← docA bridges X and Y
          (B, Y), (B, Z)     ← docB bridges Y and Z
          (C, X)             ← docC only in X

        After merge: C should get representative min(X,Y,Z) = X
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="X"),
                Row(doc_id="A", local_representative="Y"),
                Row(doc_id="B", local_representative="Y"),
                Row(doc_id="B", local_representative="Z"),
                Row(doc_id="C", local_representative="X"),
            ]
        )
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=10)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        # All should merge to same component via transitive chain
        assert reps["A"] == reps["B"] == reps["C"]
        assert reps["A"] == "X"

    def test_singletons_get_self_representative(self, spark):
        """
        Docs that never appeared in any pair → representative_id = doc_id.
        """
        local_results = spark.createDataFrame(
            [
                Row(doc_id="A", local_representative="A"),
                Row(doc_id="B", local_representative="A"),
            ]
        )
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="lonely1"), Row(id="lonely2")])

        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        assert reps["lonely1"] == "lonely1"
        assert reps["lonely2"] == "lonely2"

    def test_output_has_all_vertices(self, spark):
        """
        Output should contain ALL vertices, not just docs with pairs.
        """
        local_results = spark.createDataFrame([Row(doc_id="A", local_representative="A")])
        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D")])

        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        result_ids = {row["doc_id"] for row in result.collect()}

        assert result_ids == {"A", "B", "C", "D"}


# =============================================================================
# End-to-end: Phase 1 + Phase 2 combined
# =============================================================================


class TestEndToEnd:
    def test_single_cluster_single_partition(self, spark):
        """
        Simple case: A~B~C all in one partition.
        Phase 1 resolves everything. Phase 2 is a no-op.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="B", doc2="C", similarity=0.85, partition_id=0),
            ]
        ).repartition(1)

        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D")])

        local_results = _run_phase1_local_union_find(pairs)
        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        # A, B, C in same component
        assert reps["A"] == reps["B"] == reps["C"]
        # D is singleton
        assert reps["D"] == "D"

    def test_cross_partition_merge_end_to_end(self, spark):
        """
        Doc A appears in partition 0 with B, and partition 1 with C.
        Phase 1 can't connect B and C. Phase 2 merges them via A.

        Partition 0: (A,B)  → A→rep1, B→rep1
        Partition 1: (A,C)  → A→rep2, C→rep2
        Phase 2: rep1 and rep2 merge because A has both
        Result: A, B, C all same representative
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="A", doc2="C", similarity=0.85, partition_id=1),
            ]
        ).repartition(2, "partition_id")

        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C")])

        local_results = _run_phase1_local_union_find(pairs)
        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=10)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        # All three should be in the same component
        assert reps["A"] == reps["B"] == reps["C"]

    def test_two_independent_clusters_across_partitions(self, spark):
        """
        Cluster 1: A~B (partition 0)
        Cluster 2: C~D (partition 1)
        No bridge between them → two separate components.
        """
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="C", doc2="D", similarity=0.9, partition_id=1),
            ]
        ).repartition(2, "partition_id")

        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D"), Row(id="E")])

        local_results = _run_phase1_local_union_find(pairs)
        result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=5)
        reps = {row["doc_id"]: row["representative_id"] for row in result.collect()}

        # Cluster 1
        assert reps["A"] == reps["B"]
        # Cluster 2
        assert reps["C"] == reps["D"]
        # Separate clusters
        assert reps["A"] != reps["C"]
        # Singleton
        assert reps["E"] == "E"

    def test_matches_graphframes_result(self, spark):
        """
        Verify two-phase UF produces same groupings as GraphFrames.
        We don't compare exact representative_ids (they may differ),
        but the groupings must be identical.
        """
        try:
            from graphframes import GraphFrame

            GraphFrame(
                spark.createDataFrame([Row(id="test")]),
                spark.createDataFrame([], "src string, dst string"),
            )
        except Exception:
            pytest.skip("GraphFrames JAR not on classpath (runs on EMR)")

        # Build test data: 3 clusters
        # Cluster 1: A-B-C (chain)
        # Cluster 2: D-E (pair)
        # Singleton: F
        pairs = spark.createDataFrame(
            [
                Row(doc1="A", doc2="B", similarity=0.9, partition_id=0),
                Row(doc1="B", doc2="C", similarity=0.85, partition_id=0),
                Row(doc1="D", doc2="E", similarity=0.9, partition_id=1),
            ]
        ).repartition(2, "partition_id")

        vertices = spark.createDataFrame([Row(id="A"), Row(id="B"), Row(id="C"), Row(id="D"), Row(id="E"), Row(id="F")])

        # ── Two-phase UF ──
        local_results = _run_phase1_local_union_find(pairs)
        uf_result = _run_phase2_global_transitivity_closure(spark, local_results, vertices, max_iterations=10)
        uf_reps = {row["doc_id"]: row["representative_id"] for row in uf_result.collect()}

        # ── GraphFrames ──
        edges = pairs.select("doc1", "doc2")
        edges_fwd = edges.selectExpr("doc1 as src", "doc2 as dst")
        edges_bwd = edges.selectExpr("doc2 as src", "doc1 as dst")
        all_edges = edges_fwd.union(edges_bwd)

        spark.sparkContext.setCheckpointDir("/tmp/test-graphframes-checkpoints")
        g = GraphFrame(vertices, all_edges)
        gf_components = g.connectedComponents()
        # Map component Long to representative string (MIN id per component)
        gf_components.createOrReplaceTempView("gf_components")
        gf_result = spark.sql("""
            SELECT c.id AS doc_id, r.representative_id
            FROM gf_components c
            JOIN (
                SELECT component, MIN(id) AS representative_id
                FROM gf_components
                GROUP BY component
            ) r ON c.component = r.component
        """)
        gf_reps = {row["doc_id"]: row["representative_id"] for row in gf_result.collect()}

        # ── Compare groupings ──
        # Build sets of docs per representative for both approaches
        def get_groups(reps_dict):
            groups = {}
            for doc_id, rep in reps_dict.items():
                groups.setdefault(rep, set()).add(doc_id)
            return set(frozenset(g) for g in groups.values())

        uf_groups = get_groups(uf_reps)
        gf_groups = get_groups(gf_reps)

        assert uf_groups == gf_groups, f"Groupings differ!\nTwo-phase UF: {uf_groups}\nGraphFrames:  {gf_groups}"
