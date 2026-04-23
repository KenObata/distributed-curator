import logging
from collections.abc import Iterator

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

try:
    from .spark_utils import get_checkpoint_dir, set_spark_context
except Exception:
    from spark_utils import get_checkpoint_dir, set_spark_context

logger = logging.getLogger(__name__)

class Phase2GlobalTransitivityClosureQuery:
    """
    This class is called from _run_phase2_global_transitivity_closure()
    Each method is one logical step of Phase 2.
    Each method returns a DataFrame that can be tested independently.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def multiple_reps_edges_query(self, local_results: DataFrame) -> tuple[DataFrame, int]:
        # Step 2-1+2 combined:
        # - Step 2-1: For each doc_id, collect all distinct local_reps
        # - Step 2-2: Build transitive graph from docs with multiple representatives
        # For a doc with reps [A, B, C], emit edges: (A,B), (A,C)
        # Using the first element as anchor is sufficient for connectivity
        # Note: collect_set dedups
        local_results.createOrReplaceTempView("local_results")
        # src is arbitary.
        multiple_reps_edges = self.spark.sql("""
            SELECT DISTINCT reps[0] AS src, dst
            FROM (
                SELECT doc_id, collect_set(local_representative) AS reps
                FROM local_results
                GROUP BY doc_id
                HAVING size(collect_set(local_representative)) > 1
            ) multi_rep_docs
            LATERAL VIEW explode(slice(reps, 2, 100)) t AS dst
        """).persist(StorageLevel.DISK_ONLY)
        doc_with_multiple_local_representatives_count = multiple_reps_edges.count()
        logger.info(f"doc_with_multiple_local_representatives: {doc_with_multiple_local_representatives_count} edges")
        return multiple_reps_edges, doc_with_multiple_local_representatives_count

    def initialize_local_representative_component_columns(self, local_results: DataFrame) -> DataFrame:
        """
        Args:
        - local_results: DataFrame
            - output from Phase1. (doc_id: str, local_representative: str)
            - local_representative is not array yet. Array happens in Step2-1,2-2
        returns:
        - DataFrame: size of local_results
        """
        local_results.createOrReplaceTempView("local_results")
        rep_components = self.spark.sql("""
            SELECT DISTINCT local_representative, local_representative AS component
            FROM local_results
        """)
        return rep_components

    def propagate_transitive_closure_one_iteration(
        self, initial_components: DataFrame, multiple_reps_edges: DataFrame
    ) -> DataFrame:
        """
        returns:
        - DataFrame: size of DISTINCT local_representative from
         initial_components (= local_results) and multiple_reps_edges
        """
        # Propagate minimum component through meta-edges (both directions)

        initial_components.createOrReplaceTempView("initial_components")
        multiple_reps_edges.createOrReplaceTempView("multiple_reps_edges")

        new_rep_components = self.spark.sql("""
            SELECT local_representative, MIN(component) AS component
            FROM (
                SELECT local_representative, component
                FROM initial_components

                UNION ALL

                SELECT multi.dst AS local_representative, init.component
                FROM multiple_reps_edges multi
                JOIN initial_components init
                  ON multi.src = init.local_representative

                UNION ALL

                SELECT multi.src AS local_representative, init.component
                FROM multiple_reps_edges multi
                JOIN initial_components init
                  ON multi.dst = init.local_representative
            )
            GROUP BY local_representative
        """)
        return new_rep_components

    def count_changed_edge_for_convergence(self, new_rep_components: DataFrame) -> tuple[int, int]:
        """
        Check convergence - if still componets changed that means no converged yet
        We need hash_sum and distinct_components.
        - distinct_components because hash can have a collision
        - distinct_components standalone cannot detect convergence
          when a doc change a parent component but such parent component
          is already counted by other doc.
        """
        new_rep_components.createOrReplaceTempView("new_rep_components")
        row = self.spark.sql("""
            SELECT
                SUM(hash(component)) AS hash_sum,
                COUNT(DISTINCT component) AS distinct_components
            FROM new_rep_components
        """).collect()[0]
        return row["hash_sum"], row["distinct_components"]

    def map_final_doc_idto_global_representative(
        self, vertices: DataFrame, local_results: DataFrame, rep_components: DataFrame
    ) -> DataFrame:
        # local_results (doc_id, local_representative) is a subset of input_df that exceeded similarity score
        # and rep_components = (local_representative, root)

        # Rename to avoid column ID collision
        local_results_renamed = local_results.withColumnRenamed("local_representative", "lr_local_representative")
        rep_components_renamed = rep_components.withColumnRenamed("local_representative", "rc_local_representative")

        vertices.createOrReplaceTempView("vertices")  # all docs from input_df
        local_results_renamed.createOrReplaceTempView(
            "local_results_renamed"
        )  # candidate pair: (doc1, doc2, 0.9, partition1)
        rep_components_renamed.createOrReplaceTempView(
            "rep_components_renamed"
        )  # temp checkpoint of (doc_id, local_results)

        result = self.spark.sql("""
            SELECT
                v.id AS doc_id,
                COALESCE(g.representative_id, v.id) AS representative_id
            FROM vertices v
            LEFT JOIN (
                SELECT lr.doc_id, MIN(rc.component) AS representative_id
                FROM local_results_renamed lr
                JOIN rep_components_renamed rc
                ON lr.lr_local_representative = rc.rc_local_representative
                GROUP BY lr.doc_id
            ) g ON v.id = g.doc_id
        """)
        return result


def run_phase2_global_union_find(
    spark: SparkSession, multiple_reps_edges: DataFrame, local_results: DataFrame
) -> DataFrame:
    """
    Step 2-3: converge transitive graph
    this function runs UnionFind on a single executor
    because global UF requires all data in the same partition.

    Args:
    - multiple_reps_edges: (local_representative, component)
        - this df only has docs with local_representative that appeared more than one
        - ex) For a doc with reps [A, B, C], emit edges: (A,B), (A,C)

    Return:
    - DataFrame: size of DISTINCT local_representative doc.
      multiple_reps_edges is just LEFT JOIN.
    """
    # Step 1: prepare bridge table for mapping (URL -> Long)
    # UNION distinct src and dst nodes
    multiple_reps_edges.createOrReplaceTempView("multiple_reps_edges")
    set_spark_context(spark, "Step 5 Phase 2", "Build node ID mapping for integer-encoded UF")

    # Use monotonically_increasing_id() for node_id, because any hash() cause collision at scale.
    # monotonically_increasing_id() is non-deterministic (depends on partition_id at
    # task execution time). If node_mapping is recomputed (cache eviction, task retry,
    # or re-reference in a different query plan), it produces DIFFERENT node_ids.
    #
    # Step 5 below references node_mapping TWICE (as `nm` and `cm` in JOINs), which
    # triggers plan re-evaluation even with persist().
    #
    # checkpoint() writes the materialized result to HDFS and TRUNCATES the lineage,
    # so monotonically_increasing_id() can never be re-evaluated. IDs are stable.
    checkpoint_dir = get_checkpoint_dir(spark=spark, name="phase2-checkpoint-single-pass")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    node_mapping = spark.sql("""
        SELECT node, monotonically_increasing_id() AS node_id
        FROM (
            SELECT src AS node FROM multiple_reps_edges
            UNION
            SELECT dst AS node FROM multiple_reps_edges
        )
    """).checkpoint()
    node_mapping.createOrReplaceTempView("node_mapping")
    node_count = node_mapping.count()
    logger.info(f"Node ID mapping: {node_count} unique nodes encoded to Longs (does not include singleton.)")

    # Step 2: actual convert - Encode edges as (src_id: Long, dst_id: Long)
    multiple_reps_edges_converted = spark.sql("""
        SELECT sm.node_id AS src, dm.node_id AS dst
        FROM multiple_reps_edges e
        JOIN node_mapping sm
          ON e.src = sm.node
        JOIN node_mapping dm
          ON e.dst = dm.node
    """).persist(StorageLevel.DISK_ONLY)
    multiple_reps_edges_converted.count()

    # Step 3: Run Scala UF on Long-encoded edges on driver
    jvm_helper = spark._jvm.com.unionFind.PartitionAwareUnionFindUDF
    global_union_find_result_jdf = jvm_helper.runGlobalUnionFindFromDriver(multiple_reps_edges_converted._jdf)
    global_union_find_result_df = DataFrame(global_union_find_result_jdf, spark).persist(StorageLevel.DISK_ONLY)
    global_union_find_result_df_count = global_union_find_result_df.count()
    logger.info(f"Global UF result: {global_union_find_result_df_count} nodes resolved (does not include singleton.)")

    # Step 4: Repartition UF result before join
    set_spark_context(spark, "Step 5 Phase 2", "Repartition UF result back to shuffle partitions")
    num_shuffle_partitions = int(spark.conf.get("spark.sql.shuffle.partitions", "90000"))

    global_union_find_result_df = global_union_find_result_df.repartition(num_shuffle_partitions).persist(
        StorageLevel.DISK_ONLY
    )
    global_union_find_result_df.count()
    global_union_find_result_df.createOrReplaceTempView("global_union_find_result_df")

    # Step 5: Map Long IDs back to URLs
    set_spark_context(spark, "Step 5 Phase 2", "Map Long IDs back to URL strings")
    global_union_find_result_decoded_df = spark.sql("""
        SELECT
            nm.node AS local_representative,
            cm.node AS component
        FROM global_union_find_result_df uf
        JOIN node_mapping nm
          ON uf.node_id = nm.node_id
        JOIN node_mapping cm
          ON uf.component_id = cm.node_id
    """)
    global_union_find_result_decoded_df.createOrReplaceTempView("global_union_find_result_decoded_df")

    # Also need reps that have NO cross-partition edges (singletons in the local_results)
    # They won't appear in multiple_reps_edges but exist in local_results
    # this is to be consistent with iterative_propagate_transitive_closure_wrapper()
    set_spark_context(
        spark, "Step 5 Phase 2", "distinct_local_representative LEFT JOIN global_union_find_result_decoded_df"
    )
    local_results.createOrReplaceTempView("local_results")
    rep_components = spark.sql("""
            WITH distinct_local_representative as (
                SELECT DISTINCT local_representative
                FROM local_results
            )
            SELECT
                local.local_representative,
                COALESCE(global.component, local.local_representative) AS component
            FROM distinct_local_representative local
            LEFT JOIN global_union_find_result_decoded_df global
              ON local.local_representative = global.local_representative
        """)
    rep_components = rep_components.persist(StorageLevel.DISK_ONLY)
    rep_components_count = rep_components.count()
    logger.info(
        f"Phase 2 resolved via single-pass Union-Find: {rep_components_count} local_representative, includes singleton"
    )

    node_mapping.unpersist()
    multiple_reps_edges_converted.unpersist()
    global_union_find_result_df.unpersist()
    return rep_components


def iterative_propagate_transitive_closure_wrapper(
    spark: SparkSession,
    phase2_global_transitivity_closure_query: Phase2GlobalTransitivityClosureQuery,
    max_iterations: int,
    local_results: DataFrame,
    multiple_reps_edges: DataFrame,
) -> DataFrame:
    """
    Wrapper for Step 2-3: iterative converge.

    Args:
    - spark: SparkSession
    - phase2_global_transitivity_closure_query: Phase2GlobalTransitivityClosureQuery
    - max_iterations: guardrail of convergence.
    - local_results: (doc_id: str, local_representative: str) from Phase 1
        - local_representative is not in array yet
    - multiple_reps_edges: (local_represent_X, local_represent_Y)
        - only local_representative appeared in multiple doc_id
        - For a doc with reps [A, B, C], emit edges: (A,B), (A,C)

    Return:
    - DataFrame: size of DISTINCT local_representative from
         initial_components (= local_results) and multiple_reps_edges
    """
    # Set checkpoint directory for iteration lineage truncation
    checkpoint_dir = get_checkpoint_dir(spark=spark, name="iterative-union-find-checkpoints")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    # Initialize: each representative's component = itself
    rep_components = phase2_global_transitivity_closure_query.initialize_local_representative_component_columns(
        local_results=local_results
    )

    prev_checksum = None
    prev_distinct_components = None
    for i in range(max_iterations):
        rep_components.createOrReplaceTempView("rep_components")

        # Propagate minimum component through meta-edges (both directions)
        new_rep_components = phase2_global_transitivity_closure_query.propagate_transitive_closure_one_iteration(
            initial_components=rep_components, multiple_reps_edges=multiple_reps_edges
        )
        new_rep_components = new_rep_components.checkpoint()
        new_rep_components.count()

        # Check convergence - if still componets changed that means no converged yet
        hash_sum, distinct_components = phase2_global_transitivity_closure_query.count_changed_edge_for_convergence(
            new_rep_components=new_rep_components
        )
        is_converged = hash_sum == prev_checksum and prev_distinct_components == distinct_components
        prev_checksum = hash_sum
        prev_distinct_components = distinct_components
        logger.info(
            f"Phase 2 iteration {i + 1}: hash_sum={hash_sum}, \
         distinct_components={distinct_components}, converged={is_converged}"
        )

        rep_components.unpersist()

        if is_converged:
            rep_components = new_rep_components.persist(StorageLevel.DISK_ONLY)
            logger.info(f"Phase 2 converged in {i + 1} iterations")
            break
        # Checkpoint every iteration to truncate lineage
        # Note: use checkpoint() because persist() keeps the full DAG lineage on the driver
        # which causes driver OOM
        rep_components = new_rep_components
    else:
        logger.warning(
            f"Phase 2 did NOT converge after {max_iterations} iterations. "
            f"Results may contain unresolved duplicates. Consider increasing max_iterations."
        )
    if not rep_components.is_cached:
        rep_components = rep_components.persist(StorageLevel.DISK_ONLY)
        rep_components.count()
    return rep_components


def run_phase2_global_transitivity_closure(
    spark: SparkSession,
    local_results: DataFrame,
    vertices: DataFrame,
    max_iterations: int = 50,
    use_iterative_transitive_closure: bool = False,
) -> DataFrame:
    """
    Phase 2: Merge components across partition boundaries.
    df.join().groupBy().agg()

    A doc appearing in multiple partitions may have different local_reps.
    Build meta-edges between disagreeing reps, then run label propagation
    on this much smaller graph.

    Args:
    - local_results: (doc_id, local_representative) from Phase 1
    - vertices: same as input_df containing all input doc
    - max_iterations: if this exceeds, it warns.

    Return:
    - DataFrame: (doc_id, representative_id) for all docs from input_df including singletons
      output of this function will be joined back to input_df.

    Logic:
    Step 2-1: for each doc, collect across partitions since phase1 was within a partition.
        => GROUP BY doc_id, Array(local_representative)
    Step 2-2: Build transitive graph
    Step 2-3: iterative converge

    ex) Step 2-1:
        docA → [rep_X_partition_0, rep_Y_parttion1]
        docB → [rep_Y_partition_1, rep_Z_partition_2]

        Step2-2:
        (X,Y)
        (Y,Z)
        Y: (rep_X_partition_0, rep_Y_parttion1, rep_Z_partition_2) are connected
        => must all merge

        Step2-3:
        iteration1:
            rep_X: min(X, Y)       = X
            rep_Y: min(Y, X, Z)    = X
            rep_Z: min(Z, Y)       = Y
        iteration2:
            rep_X: min(X, Y (=X))       = X
            rep_Y: min(Y (=X), X, Z (=Y))    = X
            rep_Z: min(Z(=Y), Y ( = X))       = X
    """
    vertices.createOrReplaceTempView("vertices")

    phase2_global_transitivity_closure_query = Phase2GlobalTransitivityClosureQuery(spark)

    # Step 2-1+2 combined:
    # - Step 2-1: For each doc_id, collect all distinct local_reps
    # - Step 2-2: Build transitive graph from docs with multiple representatives
    # For a doc with reps [A, B, C], emit edges: (A,B), (A,C)
    multiple_reps_edges, doc_with_multiple_local_representatives_count = (
        phase2_global_transitivity_closure_query.multiple_reps_edges_query(local_results=local_results)
    )
    multiple_reps_edges.createOrReplaceTempView("multiple_reps_edges")

    # early exit
    if doc_with_multiple_local_representatives_count == 0:
        logger.info("No cross-partition edges. All components resolved locally.")
        result = spark.sql("""
            SELECT
                v.id AS doc_id,
                COALESCE(c.representative_id, v.id) AS representative_id
            FROM vertices v
            LEFT JOIN (
                SELECT doc_id, MIN(local_representative) AS representative_id
                FROM local_results
                GROUP BY doc_id
            ) c ON v.id = c.doc_id
        """)
        multiple_reps_edges.unpersist()
        return result

    # Step 2-3: iterative converge
    if doc_with_multiple_local_representatives_count > 600 * 1000 * 1000:  # 600M edges
        use_iterative_transitive_closure = True
    logger.info(
        f"Phase 2 Step 2-3: "
        f"{'single-pass Union-Find' if not use_iterative_transitive_closure else 'iterative SQL (fallback)'} "
        f"for {doc_with_multiple_local_representatives_count} cross-partition edges"
    )
    if use_iterative_transitive_closure:
        rep_components = iterative_propagate_transitive_closure_wrapper(
            spark=spark,
            phase2_global_transitivity_closure_query=phase2_global_transitivity_closure_query,
            max_iterations=max_iterations,
            local_results=local_results,
            multiple_reps_edges=multiple_reps_edges,
        )
    else:
        rep_components = run_phase2_global_union_find(
            spark=spark, multiple_reps_edges=multiple_reps_edges, local_results=local_results
        )

    rep_components.createOrReplaceTempView("rep_components")

    # ── Final: Map every doc_id to its global representative ──
    # local_results (doc_id, local_representative) is a subset of input_df that exceeded similarity score
    # and rep_components = (local_representative, root)
    result = phase2_global_transitivity_closure_query.map_final_doc_idto_global_representative(
        vertices=vertices, local_results=local_results, rep_components=rep_components
    )

    multiple_reps_edges.unpersist()
    # Don't unpersist rep_components in this function — let caller handle it
    # rep_components.unpersist()

    return result
