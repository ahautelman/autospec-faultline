"""Geometry helpers for the refinement operators (paper section 4).

Re-exports the convex hull, the region algebra, and the half-plane partition so
the refinement package and the tests import from ``faultline.geometry``.
"""

from faultline.geometry.convex_hull import ConvexHull, convex_hull
from faultline.geometry.half_plane import half_plane_partition
from faultline.geometry.set_ops import difference, intersect

__all__ = [
    "ConvexHull",
    "convex_hull",
    "difference",
    "half_plane_partition",
    "intersect",
]
