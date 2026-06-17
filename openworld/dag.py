"""Causal DAGs as a first-class input to world models.

A directed acyclic graph of causal assumptions is a natural front-end for a
world model: nodes are variables, edges are direct causal effects. This module
ingests a DAG (the ``dagitty`` / Graphviz ``.dot`` text both tools export) and
turns it into the three things a world model needs:

  * a :class:`CausalDAG` -- a normalized graph object (nodes + tags + edges) you
    can query (parents, topological order, exposures/outcomes/latents);
  * a perception **schema** -- the observed (non-latent) variables to extract
    from a record, so the DAG declares what to perceive;
  * a runnable **world** -- the DAG compiled to a verified ``CodeTransition``
    structural causal model, where each node is recomputed from its parents in
    topological order and ``do_<node>`` actions implement Pearl's do-operator.

Everything here is stdlib-only (the zero-dependency core contract). The
``dagitty`` syntax is a small, well-defined subset::

    dag {
    "Prenatal Stress" [exposure]
    "Bayley Score" [outcome]
    "Some Latent" [latent]
    "Prenatal Stress" -> "Bayley Score"
    }

Graphviz ``digraph { a -> b; ... }`` with optional ``[attrs]`` is also accepted.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CausalDAG", "parse_dag", "dag_to_schema", "dag_to_transition_code",
    "dag_to_world", "slug",
]

_TAGS = ("exposure", "outcome", "latent", "adjusted", "selected")


def slug(name: str) -> str:
    """Turn an arbitrary DAG node name into a stable Python identifier.

    Spaces and punctuation collapse to single underscores; non-ASCII is dropped;
    a leading digit is prefixed. Deterministic so the same name always maps to
    the same state key.
    """
    s = _re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_")
    if not s:
        s = "node"
    if s[0].isdigit():
        s = "n_" + s
    return s.lower()


@dataclass
class CausalDAG:
    """A normalized causal DAG: node names mapped to tags, plus directed edges."""

    nodes: Dict[str, List[str]] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    # -- tag views ----------------------------------------------------------
    def _tagged(self, tag: str) -> List[str]:
        return [n for n, tags in self.nodes.items() if tag in tags]

    def exposures(self) -> List[str]:
        return self._tagged("exposure")

    def outcomes(self) -> List[str]:
        return self._tagged("outcome")

    def latents(self) -> List[str]:
        return self._tagged("latent")

    def observed(self) -> List[str]:
        """Nodes that are not latent (the variables one can actually measure)."""
        return [n for n in self.nodes if "latent" not in self.nodes[n]]

    # -- graph structure ----------------------------------------------------
    def parents(self, node: str) -> List[str]:
        return [a for a, b in self.edges if b == node]

    def children(self, node: str) -> List[str]:
        return [b for a, b in self.edges if a == node]

    def roots(self) -> List[str]:
        return [n for n in self.nodes if not self.parents(n)]

    def topo_order(self) -> List[str]:
        """Kahn's algorithm; raises ValueError if the graph has a cycle."""
        indeg = {n: 0 for n in self.nodes}
        for _, b in self.edges:
            indeg[b] = indeg.get(b, 0) + 1
        queue = [n for n in self.nodes if indeg[n] == 0]
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in self.children(n):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("graph is not acyclic; topological order undefined")
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topo_order()
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [{"name": n, "tags": list(t)} for n, t in self.nodes.items()],
            "edges": [[a, b] for a, b in self.edges],
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_NODE_RE = _re.compile(r'^"?([^"\[\]]+?)"?\s*\[([^\]]*)\]\s*;?\s*$')
_EDGE_RE = _re.compile(r'^"?([^"\[\]]+?)"?\s*->\s*"?([^"\[\];]+?)"?\s*(?:\[[^\]]*\])?\s*;?\s*$')


def parse_dag(text: str) -> CausalDAG:
    """Parse a ``dagitty`` / Graphviz ``.dot`` DAG into a :class:`CausalDAG`.

    Recognizes ``"Name" [tags]`` node declarations (tags like ``exposure`` /
    ``outcome`` / ``latent``; ``key=value`` attributes such as ``pos`` are
    ignored) and ``"A" -> "B"`` edges. Comments (``//`` and ``#``) and the
    ``dag {`` / ``digraph {`` / ``}`` / ``bb=...`` scaffolding are skipped.
    Nodes appearing only in edges are added with no tags.
    """
    nodes: Dict[str, List[str]] = {}
    edges: List[Tuple[str, str]] = []

    def ensure(name: str) -> None:
        nodes.setdefault(name, [])

    # strip line comments, then normalize braces/semicolons into one statement
    # per line so inline graphs (`digraph { a -> b; b -> c; }`) split correctly.
    cleaned = []
    for raw in str(text).splitlines():
        line = _re.sub(r"//.*$", "", raw)
        line = _re.sub(r"#.*$", "", line)
        cleaned.append(line)
    blob = "\n".join(cleaned).replace("{", "\n").replace("}", "\n").replace(";", "\n")

    for raw in blob.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if (low.startswith(("dag", "digraph", "graph", "strict", "bb=", "rankdir",
                            "node ", "edge ", "graph ")) and "->" not in line) \
                or line in ("{", "}") or line.endswith("{"):
            continue
        m = _EDGE_RE.match(line)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            ensure(a)
            ensure(b)
            if (a, b) not in edges:
                edges.append((a, b))
            continue
        m = _NODE_RE.match(line)
        if m:
            name = m.group(1).strip()
            attrs = m.group(2)
            tags = []
            for tok in attrs.split(","):
                tok = tok.strip()
                key = tok.split("=", 1)[0].strip().lower()
                if key in _TAGS:
                    tags.append(key)
            ensure(name)
            nodes[name] = sorted(set(nodes[name]) | set(tags))
            continue
        # a bare node name on its own line
        bare = line.rstrip(";").strip().strip('"')
        if bare and "->" not in bare and "[" not in bare:
            ensure(bare)
    return CausalDAG(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# DAG -> perception schema
# ---------------------------------------------------------------------------
def dag_to_schema(dag: CausalDAG, node_type: type = int,
                  bounds: Optional[Tuple[float, float]] = (0, 1),
                  include_latent: bool = False) -> Dict[str, Any]:
    """A perception schema keyed by node SLUG: ``{slug: (type, (lo, hi))}``.

    The DAG declares which variables to perceive from a record. Latent nodes are
    excluded by default (you cannot measure them). ``bounds=None`` yields a bare
    type per field.
    """
    names = list(dag.nodes) if include_latent else dag.observed()
    spec: Any = (node_type, tuple(bounds)) if bounds is not None else node_type
    return {slug(n): spec for n in names}


# ---------------------------------------------------------------------------
# DAG -> verified structural-causal-model transition code
# ---------------------------------------------------------------------------
def dag_to_transition_code(dag: CausalDAG,
                           weights: Optional[Dict[Tuple[str, str], float]] = None,
                           threshold: float = 0.5) -> str:
    """Compile the DAG into a verified ``transition(state, action)`` SCM.

    Each non-root node is a linear-threshold function of its parents plus an
    exogenous term ``u_<slug>`` (read from state, default 0): the node fires
    (``1``) when ``sum(w_i * parent_i) + u`` exceeds ``threshold``. Roots keep
    their state value (they are set by data or by a do-intervention). Every
    ``do_<slug>`` action overrides that node and the override PROPAGATES to its
    descendants in the same topological pass -- Pearl's do-operator.

    Linear-threshold is a transparent, auditable default; supply ``weights`` to
    encode edge strengths/signs. The emitted code is plain, sandbox-safe Python.
    """
    weights = weights or {}
    order = dag.topo_order()
    by_slug = {n: slug(n) for n in dag.nodes}
    lines = [
        "def transition(state, action):",
        "    s = dict(state)",
        "    p = action.get('params', {})",
        "    name = action['name']",
    ]
    # expose every node slug + its exogenous term as locals
    for n in dag.nodes:
        sl = by_slug[n]
        lines.append(f"    {sl} = float(s.get('{sl}', 0))")
        lines.append(f"    u_{sl} = float(s.get('u_{sl}', 0))")
    for n in order:
        sl = by_slug[n]
        parents = dag.parents(n)
        lines.append(f"    if name == 'do_{sl}' and 'value' in p:")
        lines.append(f"        {sl} = float(p['value'])")
        if parents:
            terms = " + ".join(
                f"{float(weights.get((par, n), 1.0))}*{by_slug[par]}"
                for par in parents)
            lines.append("    else:")
            lines.append(f"        {sl} = 1.0 if ({terms} + u_{sl}) > {float(threshold)} else 0.0")
        # roots with no parents: keep their current value (data / intervention)
        lines.append(f"    s['{sl}'] = {sl}")
    lines.append("    return s")
    return "\n".join(lines)


def dag_to_world(dag: CausalDAG, name: str = "dag_world",
                 initial_state: Optional[Dict[str, Any]] = None,
                 weights: Optional[Dict[Tuple[str, str], float]] = None,
                 threshold: float = 0.5):
    """Compile a DAG into a runnable :class:`~openworld.World` (a verified SCM).

    Actions are ``do_<slug>`` (one per node) plus ``observe`` (run the SCM with
    no intervention). The initial state defaults every node slug and its
    exogenous ``u_*`` term to 0; pass ``initial_state`` to seed a unit.
    """
    from .transition import CodeTransition
    from .world import World

    if not dag.is_acyclic():
        raise ValueError("cannot build a world from a cyclic graph")
    by_slug = {n: slug(n) for n in dag.nodes}
    state: Dict[str, Any] = {}
    for n in dag.nodes:
        state[by_slug[n]] = 0.0
        state[f"u_{by_slug[n]}"] = 0.0
    if initial_state:
        state.update(initial_state)
    actions = [f"do_{by_slug[n]}" for n in dag.nodes] + ["observe"]
    rules = [f"{by_slug[b]} <- f({by_slug[a]}, ...)" for a, b in dag.edges]
    return World(
        name=name,
        description=f"Structural causal model compiled from a {len(dag.nodes)}-node DAG.",
        initial_state=state,
        actions=actions,
        rules=rules or ["no edges; all nodes exogenous"],
        transition=CodeTransition(dag_to_transition_code(dag, weights, threshold)),
    )
