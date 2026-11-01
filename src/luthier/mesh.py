"""Mesh helpers: CAD import (STL / OBJ) and small structured generators.

The engine works on **mid-surface triangle meshes**, one per physical part
(top plate, back plate, ribs, ...).  Thickness, material and grain direction
are attached to the part afterwards, so the CAD file only has to carry
*shape*:

* OBJ  - one ``g``/``o`` group per part; group names become part names
* STL  - one file per part (STL has no groups)

Closed *solid* meshes (a watertight violin body, say) are not mid-surfaces.
``midsurface_from_shell_solid`` is intentionally NOT provided yet; export the
plate surfaces from your CAD package as sheets (all major packages can do
this), or see docs/DESIGN.md "Roadmap: solid -> mid-surface".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

UNIT_SCALE = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "in": 0.0254, "inch": 0.0254}


@dataclass
class SurfaceMesh:
    vertices: np.ndarray  # (n,3) metres
    faces: np.ndarray  # (m,3)

    def centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(1)

    def areas(self) -> np.ndarray:
        v = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    def normals(self) -> np.ndarray:
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(0), self.vertices.max(0)


def _load_obj(path: Path, scale: float) -> Dict[str, SurfaceMesh]:
    """Minimal OBJ reader that keeps ``g``/``o`` groups as separate parts.

    Handles ``f v``, ``f v/vt``, ``f v//vn``, ``f v/vt/vn``, negative (relative)
    indices, and fan-triangulates polygons.  Vertices are shared across groups in
    OBJ, so each part is re-indexed to the vertices it actually uses.
    """
    verts: list = []
    groups: Dict[str, list] = {}
    cur = path.stem
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok or tok[0].startswith("#"):
            continue
        if tok[0] == "v":
            verts.append([float(t) for t in tok[1:4]])
        elif tok[0] in ("g", "o"):
            cur = tok[1] if len(tok) > 1 else cur
            groups.setdefault(cur, [])
        elif tok[0] == "f":
            idx = []
            for t in tok[1:]:
                i = int(t.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)
            for k in range(1, len(idx) - 1):
                groups.setdefault(cur, []).append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, float) * scale
    parts: Dict[str, SurfaceMesh] = {}
    for name, faces in groups.items():
        if not faces:
            continue
        F = np.asarray(faces, int)
        used, inv = np.unique(F, return_inverse=True)
        parts[name] = SurfaceMesh(V[used], inv.reshape(F.shape))
    return parts


def load_parts(path: str | Path, units: str = "mm") -> Dict[str, SurfaceMesh]:
    """Load an OBJ (grouped parts) or STL/PLY/... (single part, via trimesh) into named surface meshes."""
    path = Path(path)
    scale = UNIT_SCALE[units]
    if path.suffix.lower() == ".obj":
        return _load_obj(path, scale)
    import trimesh  # optional dependency, only needed for non-OBJ formats

    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        return {str(n): SurfaceMesh(np.asarray(g.vertices, float) * scale, np.asarray(g.faces, int))
                for n, g in obj.geometry.items()}
    return {path.stem: SurfaceMesh(np.asarray(obj.vertices, float) * scale, np.asarray(obj.faces, int))}


def save_obj(parts: Dict[str, SurfaceMesh], path: str | Path, units: str = "mm") -> None:
    """Write named parts as an OBJ with one group each (round-trips with ``load_parts``)."""
    s = 1.0 / UNIT_SCALE[units]
    off = 1
    lines = []
    for name, m in parts.items():
        lines.append(f"g {name}")
        lines += [f"v {x*s:.6f} {y*s:.6f} {z*s:.6f}" for x, y, z in m.vertices]
        lines += [f"f {a+off} {b+off} {c+off}" for a, b, c in m.faces]
        off += len(m.vertices)
    Path(path).write_text("\n".join(lines) + "\n")


def rect_mesh(a: float, b: float, nx: int, ny: int, z: float = 0.0) -> SurfaceMesh:
    """Structured triangulated rectangle [0,a]x[0,b] at height z."""
    xs, ys = np.linspace(0, a, nx + 1), np.linspace(0, b, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    faces = []
    for i in range(nx):
        for j in range(ny):
            p, q, r, s = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            if (i + j) % 2 == 0:
                faces += [(p, q, r), (p, r, s)]
            else:
                faces += [(p, q, s), (q, r, s)]
    return SurfaceMesh(V, np.array(faces))
