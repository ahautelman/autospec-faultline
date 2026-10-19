"""Geometry helpers for the refinement operators (paper section 4).

Re-exports the convex hull, the region algebra, the half-plane partition, and the
region bounds so the refinement package, the environment, and the tests import
from ``faultline.geometry``.
"""

from faultline.geometry.bounds import region_bounds, sample_within
from faultline.geometry.convex_hull import ConvexHull, convex_hull
from faultline.geometry.half_plane import half_plane_partition
from faultline.geometry.serialize import predicate_from_dict, predicate_to_dict
from faultline.geometry.set_ops import difference, intersect

__all__ = [
    "ConvexHull",
    "convex_hull",
    "difference",
    "half_plane_partition",
    "intersect",
    "predicate_from_dict",
    "predicate_to_dict",
    "region_bounds",
    "sample_within",
]
