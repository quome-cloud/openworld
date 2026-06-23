"""Program representation: a sequence of primitives applied left-to-right."""
from typing import List, Tuple, Optional, Callable
from itertools import product
from .primitives import Grid, Primitive


class Program:
    def __init__(self, steps: List[Tuple[str, Primitive]]):
        self.steps = steps
        self.name = " -> ".join(s[0] for s in steps)

    def __call__(self, g: Grid) -> Optional[Grid]:
        x = g
        for _, fn in self.steps:
            x = fn(x)
            if x is None:
                return None
        return x

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"Program({self.name!r})"


def enumerate_programs(prims: dict, max_depth: int = 2):
    """Yield programs in increasing length order.

    Depth 1: each single primitive.
    Depth 2+: all combinations of that many primitives.
    """
    names = list(prims)
    # depth 1
    for n in names:
        yield Program([(n, prims[n])])
    # depth 2+
    for d in range(2, max_depth + 1):
        for combo in product(names, repeat=d):
            yield Program([(n, prims[n]) for n in combo])
