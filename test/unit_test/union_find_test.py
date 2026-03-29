"""
Unit tests for Union-Find with path compression + union by rank.
Each test visualizes the parent/rank state to build intuition.

Run: python test_union_find.py
"""

import unittest

from union_find import UnionFind


class TestUnionFindBasics(unittest.TestCase):
    def test_01_make_set_each_node_is_its_own_parent(self):
        """
        After initial_setup, each node points to itself.

        A    B    C    (three isolated nodes)
        ↓    ↓    ↓
        A    B    C

        parent = {A: A, B: B, C: C}
        rank   = {A: 0, B: 0, C: 0}
        """
        uf = UnionFind()
        uf.initial_setup("A")
        uf.initial_setup("B")
        uf.initial_setup("C")

        self.assertEqual(uf.parent, {"A": "A", "B": "B", "C": "C"})
        self.assertEqual(uf.rank, {"A": 0, "B": 0, "C": 0})

        # Every node is its own root
        self.assertEqual(uf.find("A"), "A")
        self.assertEqual(uf.find("B"), "B")
        self.assertEqual(uf.find("C"), "C")

    def test_02_simple_union_two_nodes(self):
        """
        Union(A, B): Both rank 0, so A wins (first arg with equal rank).
        B now points to A. A's rank increases to 1.

        Before:   A    B          After:   A (rank 1)
                  ↓    ↓                   ↑
                  A    B                   B (rank 0)

        parent = {A: A, B: A}
        rank   = {A: 1, B: 0}
        """
        uf = UnionFind()
        uf.initial_setup("A")
        uf.initial_setup("B")

        uf.union("A", "B")

        self.assertEqual(uf.parent["B"], "A")  # B points to A
        self.assertEqual(uf.parent["A"], "A")  # A is root
        self.assertEqual(uf.rank["A"], 1)  # A's rank increased
        self.assertEqual(uf.find("B"), "A")

    def test_03_union_by_rank_shorter_tree_goes_under_taller(self):
        """
        Union(A, B) → A is root (rank 1)
        Union(C, D) → C is root (rank 1)
        Union(A, C) → Equal rank, A wins (rank 2). C goes under A.

        Step 1:  A        Step 2:  C        Step 3:     A (rank 2)
                 ↑                 ↑                   / \\
                 B                 D                  B   C (rank 1)
                                                          ↑
                                                          D

        parent = {A: A, B: A, C: A, D: C}
        rank   = {A: 2, B: 0, C: 1, D: 0}
        """
        uf = UnionFind()
        for x in ["A", "B", "C", "D"]:
            uf.initial_setup(x)

        uf.union("A", "B")  # A(rank 1) ← B
        uf.union("C", "D")  # C(rank 1) ← D
        uf.union("A", "C")  # Equal rank → A(rank 2) ← C

        self.assertEqual(uf.parent["C"], "A")  # C goes under A
        self.assertEqual(uf.parent["D"], "C")  # D still points to C
        self.assertEqual(uf.rank["A"], 2)

        # All nodes find root A
        self.assertEqual(uf.find("A"), "A")
        self.assertEqual(uf.find("B"), "A")
        self.assertEqual(uf.find("C"), "A")
        self.assertEqual(uf.find("D"), "A")

    def test_04_union_by_rank_tall_tree_absorbs_short(self):
        """
        Build a rank-2 tree on A, then union with standalone E.
        E (rank 0) goes under A (rank 2). Rank doesn't increase.

        Before union(A, E):

             A (rank 2)         E (rank 0)
            / \\
           B   C (rank 1)
               ↑
               D

        After union(A, E):

             A (rank 2)     ← rank stays 2 (not equal ranks)
            /|\\
           B  C  E
              ↑
              D

        parent = {A: A, B: A, C: A, D: C, E: A}
        rank   = {A: 2, B: 0, C: 1, D: 0, E: 0}
        """
        uf = UnionFind()
        for x in ["A", "B", "C", "D", "E"]:
            uf.initial_setup(x)

        uf.union("A", "B")
        uf.union("C", "D")
        uf.union("A", "C")
        # Now A is rank 2

        uf.union("A", "E")  # E(rank 0) goes under A(rank 2)

        self.assertEqual(uf.parent["E"], "A")
        self.assertEqual(uf.rank["A"], 2)  # Rank unchanged (2 > 0)


class TestPathCompression(unittest.TestCase):
    def test_05_path_compression_flattens_chain(self):
        """
        Build a chain: D → C → B → A (worst case without rank-based union).
        Then find(D) compresses the entire path.

        Before find(D):

        A ← B ← C ← D    (chain of length 3)

        parent = {A: A, B: A, C: B, D: C}

        find(D) walks: D → C → B → A (root found)
        Then compresses: D → A, C → A

        After find(D):

             A
           / | \\
          B  C  D    (everything points directly to A)

        parent = {A: A, B: A, C: A, D: A}
        """
        uf = UnionFind()
        for x in ["A", "B", "C", "D"]:
            uf.initial_setup(x)

        # Manually build a chain (bypassing union to force worst case)
        uf.parent["B"] = "A"
        uf.parent["C"] = "B"
        uf.parent["D"] = "C"

        # Before compression
        self.assertEqual(uf.parent["D"], "C")  # D → C
        self.assertEqual(uf.parent["C"], "B")  # C → B
        self.assertEqual(uf.parent["B"], "A")  # B → A

        # find(D) triggers path compression
        root = uf.find("D")

        self.assertEqual(root, "A")

        # After compression: ALL nodes point directly to A
        self.assertEqual(uf.parent["D"], "A")
        self.assertEqual(uf.parent["C"], "A")
        self.assertEqual(uf.parent["B"], "A")

    def test_06_path_compression_step_by_step(self):
        """
        Trace the path compression algorithm line by line.

        Chain: E → D → C → B → A

        find(E):
          Phase 1 - Find root:
            root = E, parent[E]=D, not root → root = D
            root = D, parent[D]=C, not root → root = C
            root = C, parent[C]=B, not root → root = B
            root = B, parent[B]=A, not root → root = A
            root = A, parent[A]=A, IS root  → stop. root = A

          Phase 2 - Compress path:
            x = E, parent[E]=D ≠ A → set parent[E]=A, x = D
            x = D, parent[D]=C ≠ A → set parent[D]=A, x = C
            x = C, parent[C]=B ≠ A → set parent[C]=A, x = B
            x = B, parent[B]=A = A → stop

          Result: E, D, C, B all point to A
        """
        uf = UnionFind()
        for x in ["A", "B", "C", "D", "E"]:
            uf.initial_setup(x)

        # Build chain
        uf.parent["B"] = "A"
        uf.parent["C"] = "B"
        uf.parent["D"] = "C"
        uf.parent["E"] = "D"

        root = uf.find("E")
        self.assertEqual(root, "A")

        # Every node now directly points to A
        for node in ["B", "C", "D", "E"]:
            self.assertEqual(uf.parent[node], "A")

    def test_07_second_find_is_O1_after_compression(self):
        """
        After path compression, subsequent finds are O(1).

        Chain: D → C → B → A

        find(D) first time:  walks 3 hops, compresses path
        find(D) second time: parent[D] = A, done in 1 hop
        """
        uf = UnionFind()
        for x in ["A", "B", "C", "D"]:
            uf.initial_setup(x)
        uf.parent["B"] = "A"
        uf.parent["C"] = "B"
        uf.parent["D"] = "C"

        # First find: compresses
        uf.find("D")

        # Second find: D already points to A
        self.assertEqual(uf.parent["D"], "A")
        self.assertEqual(uf.find("D"), "A")


class TestDeduplicationScenarios(unittest.TestCase):
    def test_08_transitive_duplicates_chain(self):
        """
        Dedup scenario: A~B, B~C, C~D (transitive chain).
        All four docs should end up in the same component.

        union(A,B): A ← B               (A is root, rank 1)
        union(B,C): find(B)=A, find(C)=C → A ← C  (rank stays 1, C rank 0)
        union(C,D): find(C)=A, find(D)=D → A ← D  (rank stays 1, D rank 0)

        Final:     A (rank 1)
                 / | \\
                B  C  D

        All find() → A
        """
        uf = UnionFind()
        pairs = [("A", "B"), ("B", "C"), ("C", "D")]

        for a, b in pairs:
            uf.initial_setup(a)
            uf.initial_setup(b)
            uf.union(a, b)

        # All docs resolve to same root
        roots = {uf.find(x) for x in ["A", "B", "C", "D"]}
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots.pop(), "A")

    def test_09_two_separate_clusters(self):
        """
        Dedup scenario: Two independent duplicate groups.

        Cluster 1: doc1 ~ doc2 ~ doc3
        Cluster 2: doc4 ~ doc5

        union(doc1,doc2): doc1 ← doc2
        union(doc2,doc3): find(doc2)=doc1, find(doc3)=doc3 → doc1 ← doc3
        union(doc4,doc5): doc4 ← doc5

        Final:
           doc1         doc4
           / \\           ↑
         doc2 doc3      doc5

        Two components: {doc1, doc2, doc3} and {doc4, doc5}
        """
        uf = UnionFind()
        pairs = [("doc1", "doc2"), ("doc2", "doc3"), ("doc4", "doc5")]

        for a, b in pairs:
            uf.initial_setup(a)
            uf.initial_setup(b)
            uf.union(a, b)

        # Cluster 1
        self.assertEqual(uf.find("doc1"), "doc1")
        self.assertEqual(uf.find("doc2"), "doc1")
        self.assertEqual(uf.find("doc3"), "doc1")

        # Cluster 2
        self.assertEqual(uf.find("doc4"), "doc4")
        self.assertEqual(uf.find("doc5"), "doc4")

        # Clusters are distinct
        self.assertNotEqual(uf.find("doc1"), uf.find("doc4"))

    def test_10_merging_two_clusters(self):
        """
        Dedup scenario: Two clusters discovered independently,
        then a bridge pair connects them.

        Step 1 - Build cluster 1:     Step 2 - Build cluster 2:
             A (rank 1)                    D (rank 1)
            / \\                            ↑
           B   C                           E

        Step 3 - Bridge pair (C, E) merges clusters:
          find(C) = A (rank 1)
          find(E) = D (rank 1)
          Equal rank → A wins, D goes under A

          Final:       A (rank 2)
                      /|\\
                     B  C  D (rank 1)
                           ↑
                           E

        This is exactly what Phase 2 meta-graph merging does!
        """
        uf = UnionFind()
        for x in ["A", "B", "C", "D", "E"]:
            uf.initial_setup(x)

        # Cluster 1 (from partition 0)
        uf.union("A", "B")
        uf.union("A", "C")

        # Cluster 2 (from partition 1)
        uf.union("D", "E")

        # Verify separate clusters
        self.assertNotEqual(uf.find("A"), uf.find("D"))

        # Bridge pair discovered (cross-partition)
        uf.union("C", "E")

        # Now all in same component
        root = uf.find("A")
        for node in ["B", "C", "D", "E"]:
            self.assertEqual(uf.find(node), root)

    def test_11_duplicate_pairs_are_harmless(self):
        """
        Same pair processed multiple times (from multiple partitions).
        Union-Find handles this correctly — union of same component = no-op.

        union(A, B) first time:  A ← B
        union(A, B) second time: find(A)=A, find(B)=A, same root → return
        union(A, B) third time:  same → return
        """
        uf = UnionFind()
        uf.initial_setup("A")
        uf.initial_setup("B")

        uf.union("A", "B")
        parent_after_first = dict(uf.parent)
        rank_after_first = dict(uf.rank)

        # Duplicate unions (from other partitions)
        uf.union("A", "B")
        uf.union("B", "A")
        uf.union("A", "B")

        # State unchanged
        self.assertEqual(uf.parent, parent_after_first)
        self.assertEqual(uf.rank, rank_after_first)

    def test_12_star_topology_all_similar_to_one_doc(self):
        """
        One hub document similar to many others.
        Common in Common Crawl (boilerplate page duplicated across domains).

        union(hub, d1): hub ← d1   rank 1
        union(hub, d2): hub ← d2   rank stays 1 (d2 rank 0 < hub rank 1)
        union(hub, d3): hub ← d3   same
        union(hub, d4): hub ← d4   same

        Final:        hub (rank 1)
                    / | | \\
                  d1 d2 d3 d4

        All docs point to hub. Perfectly flat — no compression needed.
        """
        uf = UnionFind()
        docs = ["hub", "d1", "d2", "d3", "d4"]
        for d in docs:
            uf.initial_setup(d)

        for d in ["d1", "d2", "d3", "d4"]:
            uf.union("hub", d)

        for d in docs:
            self.assertEqual(uf.find(d), "hub")

        # Tree is already flat — rank is only 1
        self.assertEqual(uf.rank["hub"], 1)


class TestEdgeCases(unittest.TestCase):
    def test_13_single_node_no_pairs(self):
        """Singleton document with no duplicates."""
        uf = UnionFind()
        uf.initial_setup("lonely_doc")

        self.assertEqual(uf.find("lonely_doc"), "lonely_doc")

    def test_14_self_pair(self):
        """A document paired with itself (shouldn't happen but handle gracefully)."""
        uf = UnionFind()
        uf.initial_setup("A")
        uf.union("A", "A")

        self.assertEqual(uf.find("A"), "A")
        self.assertEqual(uf.rank["A"], 0)  # No rank increase

    def test_15_large_chain_compression(self):
        """
        Worst-case chain of 100 nodes.
        Without path compression: find() = O(n) every time.
        With path compression: first find() = O(n), subsequent = O(1).

        Before: 0 ← 1 ← 2 ← 3 ← ... ← 99  (chain)
        After find(99): 0 ← 1, 0 ← 2, 0 ← 3, ..., 0 ← 99  (flat)
        """
        uf = UnionFind()
        for i in range(100):
            uf.initial_setup(str(i))

        # Build chain: parent[1]=0, parent[2]=1, ..., parent[99]=98
        for i in range(1, 100):
            uf.parent[str(i)] = str(i - 1)

        # find(99) compresses the entire chain
        root = uf.find("99")
        self.assertEqual(root, "0")

        # Every node now points directly to "0"
        for i in range(100):
            self.assertEqual(uf.parent[str(i)], "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
