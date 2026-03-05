# cell_04g_geometry_prover.py
"""
CTRL-MATH AIMO3 — Geometry Prover
(after cell_04f_geometry.py)

Integrates:
  - SymPy.geometry (Point, Line, Circle, Triangle)
  - AlphaGeometryRE-style rules engine (concyclic, collinear, angle chase)
  - Exposes: geometry_tool.prove_concyclic(), add_point(), solve()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import sympy
    from sympy.geometry import (
        Point as SympyPoint,
        Line as SympyLine,
        Circle as SympyCircle,
        Triangle as SympyTriangle,
        Segment as SympySegment,
        Polygon as SympyPolygon,
    )
    HAS_SYMPY_GEO = True
except ImportError:
    HAS_SYMPY_GEO = False

try:
    from cell_04_transform_engine import TransformResult
except ImportError:
    @dataclass
    class TransformResult:  # type: ignore[no-redef]
        solved: bool
        answer: Any
        reduced_state: Any
        certificate: Dict[str, Any]
        transform_name: str

try:
    from cell_04f_geometry import shoelace_exact, boundary_lattice_points, picks_theorem
except ImportError:
    shoelace_exact = None
    boundary_lattice_points = None
    picks_theorem = None


# ── AlphaGeometryRE Rules Engine ──────────────────────────────────────────────

@dataclass
class GeoPoint:
    """A named point with coordinates."""
    name: str
    x: float
    y: float

    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "GeoPoint") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class GeoFact:
    """A geometric fact (predicate)."""
    predicate: str      # "collinear", "concyclic", "parallel", "perpendicular", etc.
    points: List[str]   # names of involved points
    value: Any = None   # optional numeric value (angle, ratio, etc.)


class AlphaGeometryRE:
    """
    Rule-based geometry verifier inspired by AlphaGeometry.

    Maintains a database of points and derived facts. Applies deduction
    rules to prove geometric predicates (concyclic, collinear, etc.).
    """

    EPS = 1e-9  # numerical tolerance

    def __init__(self):
        self.points: Dict[str, GeoPoint] = {}
        self.facts: List[GeoFact] = []

    def add_point(self, name: str, x: float, y: float) -> GeoPoint:
        """Register a named point."""
        p = GeoPoint(name=name, x=x, y=y)
        self.points[name] = p
        return p

    def add_midpoint(self, name: str, a_name: str, b_name: str) -> GeoPoint:
        """Add the midpoint of segment AB."""
        a, b = self.points[a_name], self.points[b_name]
        mx = (a.x + b.x) / 2
        my = (a.y + b.y) / 2
        return self.add_point(name, mx, my)

    def add_point_on_segment(
        self, name: str, a_name: str, b_name: str, ratio: float = 0.5
    ) -> GeoPoint:
        """Add a point dividing segment AB in the given ratio (from A)."""
        a, b = self.points[a_name], self.points[b_name]
        px = a.x + ratio * (b.x - a.x)
        py = a.y + ratio * (b.y - a.y)
        return self.add_point(name, px, py)

    def _cross(self, o: GeoPoint, a: GeoPoint, b: GeoPoint) -> float:
        """Cross product (a - o) × (b - o)."""
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    # ── Predicate checks ──────────────────────────────────────────────────────

    def check_collinear(self, *names: str) -> bool:
        """Check if 3+ points are collinear."""
        if len(names) < 3:
            return True
        pts = [self.points[n] for n in names]
        for i in range(2, len(pts)):
            if abs(self._cross(pts[0], pts[1], pts[i])) > self.EPS:
                return False
        return True

    def check_concyclic(self, a_name: str, b_name: str, c_name: str, d_name: str) -> bool:
        """
        Check if four points A, B, C, D are concyclic using the determinant test.

        Four points are concyclic iff the following determinant is zero:
        | ax-dx  ay-dy  (ax-dx)^2+(ay-dy)^2 |
        | bx-dx  by-dy  (bx-dx)^2+(by-dy)^2 | = 0
        | cx-dx  cy-dy  (cx-dx)^2+(cy-dy)^2 |
        """
        a, b, c, d = (self.points[n] for n in (a_name, b_name, c_name, d_name))
        ax, ay = a.x - d.x, a.y - d.y
        bx, by = b.x - d.x, b.y - d.y
        cx, cy = c.x - d.x, c.y - d.y

        det = (
            ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
            - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
            + (ax * ax + ay * ay) * (bx * cy - by * cx)
        )
        return abs(det) < self.EPS

    def check_perpendicular(self, a: str, b: str, c: str, d: str) -> bool:
        """Check if line AB ⊥ line CD."""
        pa, pb = self.points[a], self.points[b]
        pc, pd = self.points[c], self.points[d]
        dx1, dy1 = pb.x - pa.x, pb.y - pa.y
        dx2, dy2 = pd.x - pc.x, pd.y - pc.y
        dot = dx1 * dx2 + dy1 * dy2
        return abs(dot) < self.EPS

    def check_parallel(self, a: str, b: str, c: str, d: str) -> bool:
        """Check if line AB ∥ line CD."""
        pa, pb = self.points[a], self.points[b]
        pc, pd = self.points[c], self.points[d]
        dx1, dy1 = pb.x - pa.x, pb.y - pa.y
        dx2, dy2 = pd.x - pc.x, pd.y - pc.y
        cross = dx1 * dy2 - dy1 * dx2
        return abs(cross) < self.EPS

    def angle_at(self, vertex: str, a: str, b: str) -> float:
        """Compute angle ∠AVB in radians."""
        v, pa, pb = self.points[vertex], self.points[a], self.points[b]
        dx1, dy1 = pa.x - v.x, pa.y - v.y
        dx2, dy2 = pb.x - v.x, pb.y - v.y
        dot = dx1 * dx2 + dy1 * dy2
        cross = dx1 * dy2 - dy1 * dx2
        return math.atan2(abs(cross), dot)

    def distance(self, a: str, b: str) -> float:
        """Distance between two named points."""
        return self.points[a].distance_to(self.points[b])

    # ── Circumcircle and related ──────────────────────────────────────────────

    def circumcircle(self, a: str, b: str, c: str) -> Optional[Tuple[float, float, float]]:
        """
        Compute circumcircle of triangle ABC.
        Returns (cx, cy, radius) or None if degenerate.
        """
        pa, pb, pc = self.points[a], self.points[b], self.points[c]
        ax, ay = pa.x, pa.y
        bx, by = pb.x, pb.y
        cx, cy = pc.x, pc.y

        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < self.EPS:
            return None

        ux = ((ax * ax + ay * ay) * (by - cy) +
              (bx * bx + by * by) * (cy - ay) +
              (cx * cx + cy * cy) * (ay - by)) / D
        uy = ((ax * ax + ay * ay) * (cx - bx) +
              (bx * bx + by * by) * (ax - cx) +
              (cx * cx + cy * cy) * (bx - ax)) / D
        r = math.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
        return (ux, uy, r)

    # ── Deduction rules (AlphaGeometry-style) ─────────────────────────────────

    def prove_concyclic(self, a: str, b: str, c: str, d: str) -> Dict[str, Any]:
        """
        Attempt to prove concyclicity of A, B, C, D.

        Tries:
          1. Direct determinant test
          2. Inscribed angle theorem: ∠ACB = ∠ADB
          3. SymPy.geometry verification (if available)
        """
        result = {"proved": False, "method": None, "details": {}}

        # Method 1: Direct determinant test
        if all(n in self.points for n in (a, b, c, d)):
            if self.check_concyclic(a, b, c, d):
                result["proved"] = True
                result["method"] = "determinant"
                return result

        # Method 2: Inscribed angle theorem
        if all(n in self.points for n in (a, b, c, d)):
            angle_acb = self.angle_at(c, a, b)
            angle_adb = self.angle_at(d, a, b)
            if abs(angle_acb - angle_adb) < self.EPS:
                result["proved"] = True
                result["method"] = "inscribed_angle"
                result["details"] = {
                    "angle_ACB": math.degrees(angle_acb),
                    "angle_ADB": math.degrees(angle_adb),
                }
                return result

        # Method 3: SymPy.geometry
        if HAS_SYMPY_GEO and all(n in self.points for n in (a, b, c, d)):
            try:
                pts = [SympyPoint(self.points[n].x, self.points[n].y)
                       for n in (a, b, c)]
                circ = SympyCircle(*pts)
                d_pt = SympyPoint(self.points[d].x, self.points[d].y)
                dist_to_center = d_pt.distance(circ.center)
                if abs(float(dist_to_center) - float(circ.radius)) < self.EPS:
                    result["proved"] = True
                    result["method"] = "sympy_circle"
                    return result
            except Exception:
                pass

        return result

    def prove_collinear(self, *names: str) -> Dict[str, Any]:
        """Attempt to prove collinearity of named points."""
        result = {"proved": False, "method": None}
        if all(n in self.points for n in names):
            if self.check_collinear(*names):
                result["proved"] = True
                result["method"] = "cross_product"
        return result

    # ── Triangle properties ───────────────────────────────────────────────────

    def triangle_area(self, a: str, b: str, c: str) -> float:
        """Area of triangle ABC."""
        pa, pb, pc = self.points[a], self.points[b], self.points[c]
        return abs(self._cross(pa, pb, pc)) / 2

    def triangle_properties(self, a: str, b: str, c: str) -> Dict[str, Any]:
        """Compute comprehensive triangle properties."""
        props: Dict[str, Any] = {}
        pa, pb, pc = self.points[a], self.points[b], self.points[c]

        # Side lengths
        ab = pa.distance_to(pb)
        bc = pb.distance_to(pc)
        ca = pc.distance_to(pa)
        props["sides"] = {"AB": ab, "BC": bc, "CA": ca}

        # Area
        s = (ab + bc + ca) / 2
        area = math.sqrt(max(0, s * (s - ab) * (s - bc) * (s - ca)))
        props["area"] = area
        props["perimeter"] = ab + bc + ca

        # Angles (degrees)
        props["angles"] = {
            "A": math.degrees(self.angle_at(a, b, c)),
            "B": math.degrees(self.angle_at(b, a, c)),
            "C": math.degrees(self.angle_at(c, a, b)),
        }

        # Circumcircle
        cc = self.circumcircle(a, b, c)
        if cc:
            props["circumradius"] = cc[2]
            props["circumcenter"] = (cc[0], cc[1])

        # Incircle
        if area > 0:
            props["inradius"] = area / s

        return props


# ── GeometryTool (high-level API for LLM sandbox) ────────────────────────────

class GeometryTool:
    """
    High-level geometry tool exposed to the TIR sandbox.

    Provides:
      - add_point(name, x, y)
      - prove_concyclic(A, B, C, D)
      - prove_collinear(A, B, C, ...)
      - triangle_area(A, B, C)
      - distance(A, B)
      - angle(vertex, A, B)
      - circumcircle(A, B, C)
      - solve(problem_text) → TransformResult
    """

    def __init__(self):
        self.engine = AlphaGeometryRE()

    def reset(self):
        """Clear all points and facts."""
        self.engine = AlphaGeometryRE()

    def add_point(self, name: str, x: float, y: float) -> GeoPoint:
        """Add a named point."""
        return self.engine.add_point(name, x, y)

    def add_midpoint(self, name: str, a: str, b: str) -> GeoPoint:
        """Add midpoint of segment AB."""
        return self.engine.add_midpoint(name, a, b)

    def add_point_on_segment(
        self, name: str, a: str, b: str, ratio: float = 0.5
    ) -> GeoPoint:
        """Add point dividing AB in given ratio from A."""
        return self.engine.add_point_on_segment(name, a, b, ratio)

    def prove_concyclic(self, a: str, b: str, c: str, d: str) -> Dict[str, Any]:
        """Prove four points are concyclic."""
        return self.engine.prove_concyclic(a, b, c, d)

    def prove_collinear(self, *names: str) -> Dict[str, Any]:
        """Prove points are collinear."""
        return self.engine.prove_collinear(*names)

    def triangle_area(self, a: str, b: str, c: str) -> float:
        """Area of triangle ABC."""
        return self.engine.triangle_area(a, b, c)

    def triangle_properties(self, a: str, b: str, c: str) -> Dict[str, Any]:
        """Comprehensive triangle properties."""
        return self.engine.triangle_properties(a, b, c)

    def distance(self, a: str, b: str) -> float:
        """Distance between two points."""
        return self.engine.distance(a, b)

    def angle(self, vertex: str, a: str, b: str) -> float:
        """Angle ∠AVB in degrees."""
        return math.degrees(self.engine.angle_at(vertex, a, b))

    def circumcircle(self, a: str, b: str, c: str) -> Optional[Tuple[float, float, float]]:
        """Circumcircle (cx, cy, radius)."""
        return self.engine.circumcircle(a, b, c)

    def check_perpendicular(self, a: str, b: str, c: str, d: str) -> bool:
        """Check AB ⊥ CD."""
        return self.engine.check_perpendicular(a, b, c, d)

    def check_parallel(self, a: str, b: str, c: str, d: str) -> bool:
        """Check AB ∥ CD."""
        return self.engine.check_parallel(a, b, c, d)

    def solve(self, problem_text: str) -> "TransformResult":
        """
        Attempt to solve a geometry problem from text.

        Parses keywords to identify the geometry sub-type and dispatches
        to the appropriate solver.
        """
        text_lower = problem_text.lower()
        certificate: Dict[str, Any] = {}

        # Detect concyclic/collinear keywords
        if "concyclic" in text_lower or "lie on a circle" in text_lower:
            certificate["detected"] = "concyclic_verification"
            certificate["note"] = "Requires point coordinates to verify"
            return TransformResult(
                solved=False, answer=None, reduced_state=None,
                certificate=certificate,
                transform_name="geometry_prover_v1",
            )

        if "triangle" in text_lower and "area" in text_lower:
            certificate["detected"] = "triangle_area"
            return TransformResult(
                solved=False, answer=None, reduced_state=None,
                certificate=certificate,
                transform_name="geometry_prover_v1",
            )

        return TransformResult(
            solved=False, answer=None, reduced_state=None,
            certificate={"detected": "unknown_geometry"},
            transform_name="geometry_prover_v1",
        )

    def generate_mutations(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Generate n mutated geometry problems for synthetic training data.

        Creates random triangle configurations and derives problems
        (area, angle, concyclicity).
        """
        import random
        mutations = []
        for i in range(n):
            # Random triangle
            coords = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(3)]
            self.reset()
            self.add_point("A", coords[0][0], coords[0][1])
            self.add_point("B", coords[1][0], coords[1][1])
            self.add_point("C", coords[2][0], coords[2][1])

            area = self.triangle_area("A", "B", "C")
            props = self.triangle_properties("A", "B", "C")

            # Add a point on segment AB
            ratio = random.uniform(0.1, 0.9)
            self.add_point_on_segment("D", "A", "B", ratio)

            mutations.append({
                "id": f"geo_mut_{i}",
                "type": "triangle",
                "coordinates": coords,
                "area": area,
                "properties": props,
                "ratio": ratio,
                "problem": (
                    f"In triangle ABC with vertices A({coords[0][0]:.2f}, {coords[0][1]:.2f}), "
                    f"B({coords[1][0]:.2f}, {coords[1][1]:.2f}), "
                    f"C({coords[2][0]:.2f}, {coords[2][1]:.2f}), "
                    f"find the area of the triangle."
                ),
                "answer": round(area, 6),
            })

        return mutations


# ── Module-level singleton ────────────────────────────────────────────────────

geometry_tool = GeometryTool()
