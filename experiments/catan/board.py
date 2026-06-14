"""7-hex Catan board: topology, adjacency, and production tables.

Board layout (hex IDs, flat-top orientation):

        [0][1][2]
       [3][4][5][6]   ← center row (4 = center hex)

No wait — 7 hexes in triangular arrangement: 1 center + 6 surrounding.
Axial coordinates (q, r) with center at (0,0):

    Surrounding hexes in order:
      0=(+1, 0)  E
      1=(+1,-1)  NE
      2=( 0,-1)  NW
      3=(-1, 0)  W
      4=(-1,+1)  SW
      5=( 0,+1)  SE
    Center:
      6=( 0, 0)

Vertex IDs: each hex has 6 corners; shared corners are merged.
7-hex ring: 6 outer hexes (6×4 non-shared outer vertices) + center sharing all corners.

Pre-computed adjacency tables below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

# ── Resource types ────────────────────────────────────────────────────────────

STONE = "Stone"
WOOD = "Wood"
GRAIN = "Grain"
RESOURCES = (STONE, WOOD, GRAIN)

# ── Board definition ──────────────────────────────────────────────────────────
# 7 hexes: index 0-6.  Surrounding hexes 0-5, center hex 6.
# Resource assignment per design doc:
#   Stone: 3 hexes  (tokens: 3, 9, 11)
#   Wood:  2 hexes  (tokens: 5, 8)
#   Grain: 2 hexes  (tokens: 4, 6)

HEX_RESOURCE: Dict[int, str] = {
    0: STONE,
    1: WOOD,
    2: GRAIN,
    3: STONE,
    4: WOOD,
    5: GRAIN,
    6: STONE,
}

HEX_TOKEN: Dict[int, int] = {
    0: 3,
    1: 5,
    2: 4,
    3: 9,
    4: 8,
    5: 6,
    6: 11,
}

# ── Vertex topology ───────────────────────────────────────────────────────────
# For a 7-hex board (1 center + 6 ring hexes), there are 18 unique vertices.
# We number them 0-17.
#
# Each hex has 6 vertices.  Vertices shared between hexes are identified by the
# same ID.  Layout uses pointy-top hexes; axial grid:
#
#   Ring hexes (hex 0-5) in CCW order starting at E:
#     hex 0: E  (q= 1, r= 0)
#     hex 1: NE (q= 1, r=-1)
#     hex 2: NW (q= 0, r=-1)
#     hex 3: W  (q=-1, r= 0)
#     hex 4: SW (q=-1, r= 1)
#     hex 5: SE (q= 0, r= 1)
#   Center hex: hex 6 (q=0, r=0)
#
# Vertex numbering (18 total):
#   Outer ring (12 outer vertices, one per corner of the 6 ring hexes that are
#   NOT shared with the center ring):
#     v0-v11  (2 outer vertices per ring hex, numbered CCW)
#   Inner ring (6 vertices shared between ring hexes and center hex):
#     v12-v17 (one per ring hex, shared with center)

# HEX_VERTICES[hex_id] = tuple of 6 vertex IDs (in CCW order from N)
HEX_VERTICES: Dict[int, Tuple[int, ...]] = {
    # ring hex 0 (E): outer verts 0,1; inner shared with hex1→v12, hex5→v17
    0: (12, 0, 1, 13, 17, 6),   # N_inner, NE_outer, SE_outer, S_inner(?), ...
    # We use a consistent pre-computed table derived from the 7-hex topology.
    # Recomputed below with explicit vertex assignment.
}

# ── Explicit vertex assignment ────────────────────────────────────────────────
# Vertices of the 7-hex board, numbered 0-17.
#
# Strategy: label vertices systematically.
# Inner ring (center hex corners) = v0..v5  (CCW from top)
# Outer ring (remaining vertices on ring hexes) = v6..v17  (2 per ring hex)
#
# Center hex (hex 6) corners: v0(N), v1(NE), v2(SE), v3(S), v4(SW), v5(NW)
#
# Ring hex k (0-5) shares 2 adjacent inner-ring corners with center hex:
#   hex k shares center-hex corners v_k and v_{(k+1)%6}
#   hex k also has 4 outer corners.  But some outer corners are shared with
#   neighbouring ring hexes.
#   Each ring hex contributes 2 NEW outer vertices (the ones not shared with
#   any other ring hex or center).
#
# So total vertices = 6 (inner) + 6×2 (outer) = 18.  ✓
#
# Ring hex k (k=0..5) vertices (CCW from "inner-close-N"):
#   Corner 0 (shared with center, "near"): v_k
#   Corner 1 (outer unique-A):             v_{6 + 2k}
#   Corner 2 (outer unique-B):             v_{6 + 2k + 1}
#   Corner 3 (shared with center, "far"):  v_{(k+1)%6}
#   Corner 4 (outer, shared with next ring hex's unique-A or prev's unique-B):
#             → shared with ring hex (k+1)%6 → that hex's v_{6+2((k+1)%6)}
#             WAIT: actually ring hexes don't share outer vertices with each
#             other in this topology. Let me recount.
#
# In a 1-ring hex arrangement (1 center + 6 ring):
#   - Each ring hex shares 2 vertices with the center hex.
#   - Adjacent ring hexes share 1 vertex with each other (the outer corner
#     between them).
#
# So per ring hex: 6 corners = 2 shared-with-center + 2 shared-with-neighbor-ring
#                              + 2 uniquely-outer = 6. ✓
#
# Total unique vertices: 6 (inner/center) + 6 (shared between adjacent rings)
#                        + 6×2=12 uniquely-outer... = 24? That's too many.
#
# Let me use the standard Catan count for small boards.
# Actually for 7 hexes (standard Catan has 19 hexes → 54 vertices), a
# single-ring arrangement has:
#   - 6 inner vertices (center hex)
#   - 6 outer-shared vertices (between adjacent ring hexes)
#   - 6×2=12 uniquely outer vertices (each ring hex contributes 2)
#   = 24 total... but the design doc says 18.
#
# Let me re-read the design: "7-hex board yields 18 settleable vertices".
# Standard formula: for a hex grid of N hexes, vertices = 2N + 2... no.
# Actually vertices = sum over hexes of unique corners.
# For 7 hexes with the sharing pattern: center has 6, each ring hex adds
# 4 new ones (shares 2 with center), but adjacent ring hexes share 1 corner
# each. 6 + 6*4 - 6*1 = 6 + 24 - 6 = 24. So 24 vertices, not 18.
#
# The design doc says 18, which might be a simplified/approximated number
# or refers to "interior" settleable vertices only. Let's just use the
# correct count (24) and trust the implementation. The exact count doesn't
# affect the research question.

# Let me implement this cleanly with axial coordinates and derived topology.

def _build_board() -> "_BoardTopology":
    """Build the 7-hex board topology from axial coordinates."""
    return _BoardTopology()


@dataclass
class _BoardTopology:
    """Pre-computed board topology for the 7-hex Catan variant."""

    # Axial coords for each hex (q, r)
    hex_coords: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # hex_id → list of vertex_ids (6 per hex, CCW from NW corner)
    hex_to_verts: Dict[int, List[int]] = field(default_factory=dict)
    # vertex_id → list of hex_ids adjacent
    vert_to_hexes: Dict[int, List[int]] = field(default_factory=dict)
    # vertex_id → list of neighboring vertex_ids (connected by a road edge)
    vert_neighbors: Dict[int, List[int]] = field(default_factory=dict)
    # edge as frozenset(v1,v2) → edge_id
    edge_ids: Dict[FrozenSet[int], int] = field(default_factory=dict)
    # edge_id → frozenset of vertex pair
    edges: Dict[int, FrozenSet[int]] = field(default_factory=dict)
    # vertex_id → list of edge_ids
    vert_to_edges: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._compute()

    def _compute(self) -> None:
        # Axial hex coords: center + 6 ring neighbors
        # Axial neighbor directions (pointy-top): E, NE, NW, W, SW, SE
        _AX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

        center = (0, 0)
        coords = [center] + [(_AX_DIRS[i][0], _AX_DIRS[i][1]) for i in range(6)]
        # hex 0 = center, hex 1-6 = ring (reordering from design: center=6, ring=0-5)
        # Per design doc: center hex = 6, ring hexes = 0-5
        # We store: hex_id → (q, r)
        ring_order = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        self.hex_coords = {i: ring_order[i] for i in range(6)}
        self.hex_coords[6] = (0, 0)

        # For each hex, compute its 6 vertex keys using cube coordinates.
        # Pointy-top vertices of hex at (q,r) in axial:
        # Vertex at direction d uses a canonical key based on the 3 adjacent hexes.
        # We use the "three hexes share a vertex" identity to assign IDs.
        #
        # For pointy-top hex at axial (q,r), the 6 corners can be indexed by
        # the pair of axial directions.  Corner i (0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW)
        # is shared by hex (q,r) and two neighbors:
        #   Corner 0 (N):  neighbors at dir NW and NE  → dirs 2,1 → hexes (q-1,r),(q+1,r-1)  (wait wrong)
        #
        # Standard approach: represent each vertex as a frozenset of the (up to 3)
        # axial hex coords that share it, then assign integer IDs.

        all_hex_axial = set(self.hex_coords.values())

        # Pointy-top hex vertex corner directions (which two neighboring hexes share it):
        # For corner k of hex H, the two adjacent hexes are H+dir[k-1] and H+dir[k]
        # where dirs are the 6 axial neighbor directions.
        _DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]  # E,SE,SW,W,NW,NE (CCW)

        def _add(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[int,int]:
            return (a[0]+b[0], a[1]+b[1])

        vert_key_to_id: Dict[FrozenSet, int] = {}
        next_vid = [0]

        def _get_vert(key: FrozenSet) -> int:
            if key not in vert_key_to_id:
                vert_key_to_id[key] = next_vid[0]
                next_vid[0] += 1
            return vert_key_to_id[key]

        self.hex_to_verts = {}
        for hid, hq in self.hex_coords.items():
            verts = []
            outer_count = 0  # disambiguate the 2 uniquely-outer corners per ring hex
            for k in range(6):
                # Corner k is shared by hex hq and its neighbors at dir[k-1] and dir[k]
                n1 = _add(hq, _DIRS[(k - 1) % 6])
                n2 = _add(hq, _DIRS[k])
                members = frozenset(c for c in (hq, n1, n2) if c in all_hex_axial)
                if len(members) == 1:
                    # Outer corner: only this hex is present.  Two adjacent outer
                    # corners of the same ring hex produce the same frozenset, so
                    # append a per-hex counter to keep them distinct.
                    key = (members, hid, outer_count)
                    outer_count += 1
                else:
                    key = members
                verts.append(_get_vert(key))
            self.hex_to_verts[hid] = verts

        # Build vert_to_hexes
        self.vert_to_hexes = {vid: [] for vid in range(next_vid[0])}
        for hid, verts in self.hex_to_verts.items():
            for vid in verts:
                if hid not in self.vert_to_hexes[vid]:
                    self.vert_to_hexes[vid].append(hid)

        # Build vertex neighbors (road edges): two vertices are neighbors if they
        # share a hex and are adjacent corners (differ by 1 in the corner index).
        vert_neighbors: Dict[int, set] = {vid: set() for vid in range(next_vid[0])}
        next_eid = [0]
        edge_set: Dict[FrozenSet[int], int] = {}

        for hid, verts in self.hex_to_verts.items():
            n = len(verts)
            for i in range(n):
                v1 = verts[i]
                v2 = verts[(i + 1) % n]
                if v1 == v2:
                    continue  # skip degenerate self-loop (shouldn't occur after key fix)
                vert_neighbors[v1].add(v2)
                vert_neighbors[v2].add(v1)
                key = frozenset((v1, v2))
                if key not in edge_set:
                    edge_set[key] = next_eid[0]
                    next_eid[0] += 1

        self.vert_neighbors = {vid: sorted(nb) for vid, nb in vert_neighbors.items()}
        self.edge_ids = edge_set
        self.edges = {eid: key for key, eid in edge_set.items()}

        # Build vert_to_edges
        self.vert_to_edges = {vid: [] for vid in range(next_vid[0])}
        for edge_key, eid in edge_set.items():
            for vid in edge_key:
                self.vert_to_edges[vid].append(eid)

    @property
    def num_vertices(self) -> int:
        return len(self.vert_to_hexes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


# Module-level singleton
BOARD = _BoardTopology()
