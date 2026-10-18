"""Depth -> color registration for the live RGB-D tracker.

The Orbbec stream is aligned in software the same way the recorder's
``AlignFilter`` does it:

    depth raw --(SDK scale)--> metric depth --(depth intrinsics)--> 3D
              --(color<-depth extrinsic)--> color 3D
              --(color intrinsics + distortion)--> color pixels

The SDK depth scale is applied **once**, before the rigid transform.  The
extrinsic translation is metric, so multiplying the aligned depth afterwards
(as an offline replay can get away with for a rough comparison) would scale
the wrong quantity and is not done here.

Output is uint16 millimetres in the color optical frame, 0 = invalid.  That
matches cuVSLAM's ``depth_scale_factor = 1000`` and is checked against the
calibrated color shape.
"""

from __future__ import annotations

import math

import numpy as np

from .calibration import CalibrationError, RigCalibration

_MIN_DEPTH_M = 0.05


class DepthRegistration:
    """Rigid depth->color registration with the factory calibration."""

    def __init__(self, calibration: RigCalibration):
        self.calibration = calibration
        cal = calibration.depth
        ys, xs = np.mgrid[0:cal.height, 0:cal.width]
        self._xn = ((xs - cal.cx) / cal.fx).astype(np.float32)
        self._yn = ((ys - cal.cy) / cal.fy).astype(np.float32)
        self._T_color_from_depth = calibration.T_color_from_depth.astype(np.float32)

    def _metric_depth(self, depth_raw: np.ndarray,
                      depth_scale_mm: float) -> np.ndarray:
        """Validate raw depth and return float32 metres in the depth frame."""
        if depth_raw is None:
            raise CalibrationError("depth frame is missing")
        raw = np.asarray(depth_raw)
        expected = self.calibration.depth_shape
        if raw.ndim != 2 or tuple(raw.shape) != expected:
            raise CalibrationError(
                f"depth shape {tuple(raw.shape)} != calibrated "
                f"{expected} (h, w)")
        try:
            scale = float(depth_scale_mm)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                f"invalid depth_scale_mm {depth_scale_mm!r}") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise CalibrationError(
                f"invalid depth_scale_mm {scale!r}; refusing to assume mm")
        # One and only one raw-unit -> millimetre conversion.
        return raw.astype(np.float32) * (scale / 1000.0)

    def align_meters(self, depth_raw: np.ndarray,
                     depth_scale_mm: float) -> np.ndarray:
        """Color-frame depth in metres (float32, 0 = invalid)."""
        cal = self.calibration
        z = self._metric_depth(depth_raw, depth_scale_mm)
        valid = z > _MIN_DEPTH_M
        out = np.zeros(cal.color_shape, np.float32)
        if not valid.any():
            return out

        zv = z[valid]
        xv = self._xn[valid] * zv
        yv = self._yn[valid] * zv
        T = self._T_color_from_depth
        xc = T[0, 0] * xv + T[0, 1] * yv + T[0, 2] * zv + T[0, 3]
        yc = T[1, 0] * xv + T[1, 1] * yv + T[1, 2] * zv + T[1, 3]
        zc = T[2, 0] * xv + T[2, 1] * yv + T[2, 2] * zv + T[2, 3]
        front = zc > _MIN_DEPTH_M
        xc, yc, zc = xc[front], yc[front], zc[front]
        if zc.size == 0:
            return out

        xn = xc / zc
        yn = yc / zc
        k1, k2, p1, p2, k3, k4, k5, k6 = cal.color.distortion
        r2 = xn * xn + yn * yn
        num = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        den = 1.0 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        radial = num / den
        tx = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        ty = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        u = cal.color.fx * (xn * radial + tx) + cal.color.cx
        v = cal.color.fy * (yn * radial + ty) + cal.color.cy
        cw, ch = cal.color.width, cal.color.height
        inside = (u >= 0) & (u < cw - 1) & (v >= 0) & (v < ch - 1)
        u, v, zc = u[inside], v[inside], zc[inside]
        if zc.size == 0:
            return out

        # Collisions keep the nearest surface (the camera is the origin).
        scatter = np.full(cal.color_shape, np.inf, np.float32)
        np.minimum.at(scatter, (np.rint(v).astype(np.int32),
                                np.rint(u).astype(np.int32)), zc)
        hit = np.isfinite(scatter)
        out[hit] = scatter[hit]
        return out

    def align(self, depth_raw: np.ndarray, depth_scale_mm: float) -> np.ndarray:
        """Color-frame depth as uint16 millimetres (0 = invalid)."""
        meters = self.align_meters(depth_raw, depth_scale_mm)
        mm = np.rint(meters * 1000.0)
        np.clip(mm, 0.0, 65535.0, out=mm)
        return mm.astype(np.uint16)
