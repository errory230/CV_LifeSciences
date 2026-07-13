"""2D projections of aligned conformers for the in-browser structure animation.

Instead of rendering a PNG per frame, we project each frame's 3D coordinates onto
the two fixed viewing angles (the same ones `render3d` uses) and store the (x, y,
depth) per atom. The browser then draws atoms/bonds live from these coordinates,
keeping the HTML self-contained but animatable.

Frames are aligned (RMSD superposition to frame 0) before projection so the
molecule does not tumble, and all frames/angles share one global scale + per-view
centre so the molecule does not pulse in size during playback.
"""

from __future__ import annotations

import numpy as np


def view_basis(elev_deg: float, azim_deg: float):
    """Orthonormal (right, up, view) basis for an (elev, azim) camera, matching
    the convention used by matplotlib's 3D axes closely enough to look alike."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, view)
    n = np.linalg.norm(right)
    right = right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    up = np.cross(view, right)
    return right, up, view


def project(xyz: np.ndarray, angles, *, decimals: int = 2) -> list:
    """Project aligned coords `(F, A, 3)` onto each angle → list per angle of
    `(F, A, 3)` [x, y, depth], globally normalized (shared xy scale, per-view
    centre; depth in [0, 1]). Returns plain nested lists (JSON-ready).
    """
    projected = []
    for elev, azim in angles:
        right, up, view = view_basis(elev, azim)
        x = xyz @ right
        y = xyz @ up
        z = xyz @ view
        projected.append(np.stack([x, y, z], axis=-1))  # (F, A, 3)

    # shared xy scale across both views so the molecule is the same size in each
    half = max(
        (np.ptp(p[:, :, :2].reshape(-1, 2), axis=0).max() for p in projected),
        default=1.0,
    ) / 2 or 1.0

    out = []
    for p in projected:
        xy = p[:, :, :2]
        ctr = (xy.reshape(-1, 2).max(0) + xy.reshape(-1, 2).min(0)) / 2  # per-view centre
        xyn = (xy - ctr) / half
        z = p[:, :, 2:3]
        zr = (z.max() - z.min()) or 1.0
        zn = (z - z.min()) / zr  # depth 0=back .. 1=front
        arr = np.concatenate([xyn, zn], axis=2).round(decimals)
        out.append(arr.tolist())
    return out


def subsample_indices(n_frames: int, max_frames: int) -> list[int]:
    """Evenly spaced frame indices, at most `max_frames` (unified stride)."""
    if n_frames <= max_frames:
        return list(range(n_frames))
    stride = int(np.ceil(n_frames / max_frames))
    return list(range(0, n_frames, stride))
