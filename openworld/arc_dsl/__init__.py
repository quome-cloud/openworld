"""ARC-AGI geometric grid DSL for program synthesis."""
from .primitives import (
    Grid, Primitive,
    rotate_90, rotate_180, rotate_270,
    flip_lr, flip_ud,
    transpose, antitranspose,
    gravity_down, gravity_up, gravity_right, gravity_left,
    crop_to_content,
    mirror_h, mirror_v,
    invert_colors,
    sort_rows,
    outline,
    make_recolor, make_translate, make_tile,
    connected_components, largest_object, symmetry_axis,
    PURE_PRIMS, get_parameterized_prims,
)
from .program import Program, enumerate_programs

__all__ = [
    "Grid", "Primitive",
    "rotate_90", "rotate_180", "rotate_270",
    "flip_lr", "flip_ud",
    "transpose", "antitranspose",
    "gravity_down", "gravity_up", "gravity_right", "gravity_left",
    "crop_to_content",
    "mirror_h", "mirror_v",
    "invert_colors",
    "sort_rows",
    "outline",
    "make_recolor", "make_translate", "make_tile",
    "connected_components", "largest_object", "symmetry_axis",
    "PURE_PRIMS", "get_parameterized_prims",
    "Program", "enumerate_programs",
]
