"""Rotation-only bundle adjustment.

Unknowns: a small rotation vector per image (applied in the camera frame on top of the metadata pose, so
component z is pure roll about the optical axis) plus a shared focal length and radial distortion k1, k2.
Residual per match: difference between the two world rays, scaled by the focal length (~ pixels).
The metadata pose acts as a weak prior, which keeps images without matches in place and fixes the gauge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .camera import Camera, ground_parallax, unproject_normalized
from .features import PairMatches


@dataclass
class BAResult:
    cams: list[Camera]
    omega_deg: np.ndarray      # (n, 3) correction applied: [about x (tilt), about y (pan), about z (roll)]
    pair_rms_before: dict
    pair_rms_after: dict
    n_matches: int
    n_outliers_removed: int


def _stack_matches(matches: list[PairMatches]):
    ii = np.concatenate([np.full(len(m.pts_i), m.i) for m in matches])
    jj = np.concatenate([np.full(len(m.pts_j), m.j) for m in matches])
    pid = np.concatenate([np.full(len(m.pts_i), k) for k, m in enumerate(matches)])
    return ii, jj, pid, np.concatenate([m.pts_i for m in matches]), np.concatenate([m.pts_j for m in matches])


def bundle_adjust(cams: list[Camera], matches: list[PairMatches], neighbors: list[tuple[int, int]] = (),
                  positions: np.ndarray | None = None, ground_h: float = 0.0, fit_positions: bool = False,
                  position_sigma_m: float = 0.3,
                  prior_sigma_deg=(3.0, 3.0, 5.0), neighbor_sigma_deg: float = 1.5, prior_weight: float = 2.0,
                  focal_sigma_rel: float = 0.03, k_sigma: float = 0.1, loss_scale_px: float = 2.0,
                  outlier_px: float = 10.0, verbose: bool = True) -> BAResult:
    """neighbors: image pairs whose corrections (in the world frame) should be similar. This lets images
    without any matches (plain sky) follow the systematic correction of the images around them.

    positions/ground_h: camera positions (n, 3) and ground plane depth; when given, rays below the horizon are
    corrected for the drone's movement between shots (see camera.ground_parallax). With fit_positions the
    positions are refined as well, with the given ones as prior (sigma position_sigma_m)."""
    n = len(cams)
    R0 = np.stack([c.R for c in cams])
    f0, cx, cy = cams[0].focal, cams[0].cx, cams[0].cy
    sig = np.radians(np.asarray(prior_sigma_deg))
    nb = np.array(neighbors, int).reshape(-1, 2)
    nb_sig = np.radians(neighbor_sigma_deg)
    fit_positions = fit_positions and positions is not None

    ii, jj, pid, Pi, Pj = _stack_matches(matches)
    n_rot = 3 * n + 3                       # rotations + focal, k1, k2
    n_params = n_rot + (3 * n if fit_positions else 0)

    def unpack(x):
        omega = x[: 3 * n].reshape(n, 3)
        pos = positions + x[n_rot:].reshape(n, 3) if fit_positions else positions
        return omega, f0 * (1.0 + x[3 * n]), x[3 * n + 1], x[3 * n + 2], pos

    def match_residuals(x, ii, jj, Pi, Pj):
        omega, f, k1, k2, pos = unpack(x)
        Rs = Rotation.from_rotvec(omega).as_matrix() @ R0
        wi = np.einsum("nk,nkl->nl", unproject_normalized(Pi, f, cx, cy, k1, k2), Rs[ii])
        wj = np.einsum("nk,nkl->nl", unproject_normalized(Pj, f, cx, cy, k1, k2), Rs[jj])
        if pos is not None:
            wi = ground_parallax(wi, pos[ii], ground_h)
            wj = ground_parallax(wj, pos[jj], ground_h)
        return (wi - wj) * f

    def residuals(x, ii, jj, Pi, Pj):
        omega = x[: 3 * n].reshape(n, 3)
        prior = (omega / sig).ravel() * prior_weight
        intr = np.array([x[3 * n] / focal_sigma_rel, x[3 * n + 1] / k_sigma, x[3 * n + 2] / k_sigma]) * prior_weight
        # exp([w]) R0 == R0 exp([R0^T w]), so R0^T w is the correction expressed in the world frame
        world = np.einsum("nkl,nk->nl", R0, omega)
        smooth = ((world[nb[:, 0]] - world[nb[:, 1]]) / nb_sig).ravel() * prior_weight
        parts = [match_residuals(x, ii, jj, Pi, Pj).ravel(), prior, intr, smooth]
        if fit_positions:
            parts.append(x[n_rot:] / position_sigma_m * prior_weight)
        return np.concatenate(parts)

    def sparsity(ii, jj):
        # each match row depends on both images' rotations (and positions) and the shared intrinsics;
        # priors are diagonal
        m = len(ii)
        blocks = [3 * ii[:, None] + np.arange(3), 3 * jj[:, None] + np.arange(3),
                  np.broadcast_to(3 * n + np.arange(3), (m, 3))]
        if fit_positions:
            blocks += [n_rot + 3 * ii[:, None] + np.arange(3), n_rot + 3 * jj[:, None] + np.arange(3)]
        cols = np.concatenate(blocks, axis=1)
        match_rows = np.repeat(np.arange(3 * m), cols.shape[1])
        match_cols = np.repeat(cols, 3, axis=0).ravel()
        prior_idx = np.arange(3 * n + 3)
        base = 3 * m + 3 * n + 3
        nb_rows = np.repeat(base + np.arange(3 * len(nb)), 6)
        nb_cols = np.repeat(np.concatenate([3 * nb[:, :1] + np.arange(3), 3 * nb[:, 1:] + np.arange(3)], axis=1), 3, axis=0).ravel()
        rows = [match_rows, 3 * m + prior_idx, nb_rows]
        cols = [match_cols, prior_idx, nb_cols]
        n_rows = base + 3 * len(nb)
        if fit_positions:
            rows.append(n_rows + np.arange(3 * n))
            cols.append(n_rot + np.arange(3 * n))
            n_rows += 3 * n
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_rows, n_params))

    def pair_rms(x, ii, jj, Pi, Pj, pid):
        err = np.linalg.norm(match_residuals(x, ii, jj, Pi, Pj), axis=1)
        return {(matches[k].i, matches[k].j): float(np.sqrt(np.mean(err[pid == k] ** 2)))
                for k in np.unique(pid)}, err

    x0 = np.zeros(n_params)
    rms_before, _ = pair_rms(x0, ii, jj, Pi, Pj, pid)

    x = x0
    removed = 0
    for rnd in range(2):
        res = least_squares(residuals, x, jac_sparsity=sparsity(ii, jj), loss="huber", f_scale=loss_scale_px,
                            x_scale="jac", method="trf", args=(ii, jj, Pi, Pj), max_nfev=200, verbose=0)
        x = res.x
        _, err = pair_rms(x, ii, jj, Pi, Pj, pid)
        if verbose:
            print(f"  BA round {rnd + 1}: {len(ii)} matches, median {np.median(err):.2f}px, "
                  f"RMS {np.sqrt(np.mean(err ** 2)):.2f}px, cost {res.cost:.0f}")
        keep = err < outlier_px
        if keep.all():
            break
        removed += int((~keep).sum())
        ii, jj, Pi, Pj, pid = ii[keep], jj[keep], Pi[keep], Pj[keep], pid[keep]

    rms_after, _ = pair_rms(x, ii, jj, Pi, Pj, pid)
    omega, f, k1, k2, pos = unpack(x)
    Rs = Rotation.from_rotvec(omega).as_matrix() @ R0
    new_cams = [Camera(c.width, c.height, f, c.cx, c.cy, Rs[k], k1, k2, c.gain, c.meta,
                       pos[k] if pos is not None else None, ground_h) for k, c in enumerate(cams)]
    return BAResult(new_cams, np.degrees(omega), rms_before, rms_after, len(ii), removed)
