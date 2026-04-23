"""
Union-Find (Disjoint Set) with path compression + union by rank.

Used by:
  - two_phase_union_find.py (Phase 1 partition-local UF)
  - test_union_find.py (unit tests)
"""


class UnionFind:
    """
    Classic Union-Find with:
      - Path compression (iterative, not recursive — avoids stack overflow)
      - Union by rank (attach shorter tree under taller)
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def initial_setup(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0

    def find(self, node: str) -> str:
        # Find root
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression: point every node on the path directly to root
        while self.parent[node] != root:
            next_parent = self.parent[node]
            self.parent[node] = root  # node's parent as root
            node = next_parent  # move next
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        # Union by rank: attach shorter tree under taller tree
        if self.rank.get(root_a, 0) < self.rank.get(root_b, 0):
            # treat a as bigger tree
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a

        # case1: A, B are equal ranks then rank increases by 1
        if self.rank.get(root_a, 0) == self.rank.get(root_b, 0):
            self.rank[root_a] = self.rank.get(root_a, 0) + 1
        # else: no increase because short tree is attached to a bigger tree's node
        # this does not increase height
