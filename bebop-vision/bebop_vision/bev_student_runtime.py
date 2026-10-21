"""TensorRT runtime for the learned master-BEV student.

The engine is exported by ``tools/export_bev_student.py`` with a static
``[2,4,H,W]`` input (near, far; RGB + depth/8) and three outputs:
``semantic [4,120,60]``, ``ego [1,120,60]``, ``coverage [1,120,60]``.

This module owns only preprocessing, TensorRT execution and postprocessing;
the training model (``bev_student.py``) is not imported on the robot.  The
postprocessed ``BevGrid`` keeps the geometric planner/app contract
(``occ``/``raw``/``coverage``) and adds the learned ``semantic``, ``wall``
and ``ego`` layers.
Preprocessing derives H/W from the engine and validates the metadata shape.

Hazard policy: ``hazard_source="model"`` (default) is the pure student.
``"union"`` adds geometric BEV hazard cells; ``"geometric"`` discards the
student hazard head and uses only the geometric source.  The held-out
staircase evaluation showed the student hazard head does not generalise
(see docs/bev-student.md), so the live service currently uses model-only
at the operator's request; the geometric override is still available.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .master_bev import (FREE, HAZARD, INFLATED, OCCUPIED, SRC_BOTH,
                         SRC_NEAR, SRC_FAR, BevGrid, load_ego_raster,
                         rasterize_ego)
from .orbbec import CONFIG_DIR, load_rig_config

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_HW = (240, 424)
SEM_FLOOR, SEM_WALL, SEM_OBSTACLE, SEM_HAZARD = 0, 1, 2, 3


class StudentRuntimeError(RuntimeError):
    pass


class BevStudentRuntime:
    """Load one TensorRT engine and run a near/far RGB-D pair."""

    def __init__(self, engine_path, meta_path=None, device="cuda",
                 hazard_source="model", hazard_threshold=0.5):
        import tensorrt as trt

        self.engine_path = str(engine_path)
        self.device = device
        self.hazard_source = hazard_source
        self.hazard_threshold = float(hazard_threshold)
        meta_path = Path(meta_path) if meta_path else self._find_meta(
            Path(engine_path))
        with open(meta_path) as f:
            self.meta = json.load(f)
        logger = trt.Logger(trt.Logger.WARNING)
        self._trt = trt
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(
                f.read())
        if self.engine is None:
            raise StudentRuntimeError(f"could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = "input"
        # Single-head engines carry ego and unknown as classes 4 and 5 of
        # the semantic output; older engines have separate sigmoid heads.
        self.single_head = bool(self.meta.get("single_head", False))
        self.output_names = (["semantic"] if self.single_head
                             else ["semantic", "ego", "coverage"])
        self.in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if (len(self.in_shape) != 4 or self.in_shape[:2] != (2, 4)
                or min(self.in_shape[2:]) <= 0):
            raise StudentRuntimeError(f"expected static [2,4,H,W], got {self.in_shape}")
        self.image_hw = self.in_shape[2:]
        meta_shape = tuple(self.meta.get("input", {}).get("shape", self.in_shape))
        if meta_shape != self.in_shape:
            raise StudentRuntimeError("metadata input shape does not match engine")
        self.in_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name))
        self.input = torch.empty(self.in_shape, device=device,
                                 dtype=self.in_dtype)
        self.outputs = {}
        for name in self.output_names:
            shp = tuple(self.engine.get_tensor_shape(name))
            self.outputs[name] = torch.empty(
                shp, device=device,
                dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)))
        self.context.set_tensor_address(self.input_name,
                                        self.input.data_ptr())
        for name, buf in self.outputs.items():
            self.context.set_tensor_address(name, buf.data_ptr())
        self.stream = torch.cuda.current_stream().cuda_stream

        self.configure_grid(self.meta.get("grid", {}),
                            load_rig_config()["robots"]["default"])
        self.last_latency_ms = None
        self.last_preprocess_ms = None
        self.runs = 0

    def configure_grid(self, grid, cfg):
        """Metric grid geometry from the engine metadata + rig config.

        Cells may be rectangular: matched-grid students (e.g. 360x640 over
        6 m x 4 m) record ``cell_x_m``/``cell_y_m`` and a null ``cell_m``;
        older square-cell metadata only carries ``cell_m``.  The static ego
        rectangle and the inflation kernel use the per-axis sizes.
        """
        ego = ((cfg.get("bev_student") or {}).get("ego") or {})
        self.ego_x = tuple(float(v) for v in ego.get(
            "x_range_m", [0.0, 0.55]))
        self.ego_y = tuple(float(v) for v in ego.get(
            "y_range_m", [-0.22, 0.22]))
        self.rows = int(grid.get("rows", 120))
        self.cols = int(grid.get("cols", 60))
        self.range_m = float(grid.get("range_m", 6.0))
        self.width_m = float(grid.get("width_m", 3.0))
        square = grid.get("cell_m")
        self.cell_x_m = float(grid.get("cell_x_m") or square
                              or self.range_m / self.rows)
        self.cell_y_m = float(grid.get("cell_y_m") or square
                              or self.width_m / self.cols)
        self.cell_m = (self.cell_x_m
                       if np.isclose(self.cell_x_m, self.cell_y_m) else None)
        x = self.range_m - (np.arange(self.rows) + 0.5) * self.cell_x_m
        y = (np.arange(self.cols) + 0.5) * self.cell_y_m - self.width_m / 2.0
        mask_file = ego.get("mask_file")
        if mask_file:
            # Learned chassis footprint (tools/bev_ego_from_data.py); the
            # rectangle above is only the fallback when the file is absent.
            path = Path(mask_file)
            if not path.is_absolute():
                path = CONFIG_DIR / path
            self.ego_static = rasterize_ego(load_ego_raster(path), x, y)
            self.ego_source = str(path)
        else:
            self.ego_static = ((x[:, None] >= self.ego_x[0])
                               & (x[:, None] <= self.ego_x[1])
                               & (y[None, :] >= self.ego_y[0])
                               & (y[None, :] <= self.ego_y[1]))
            self.ego_source = "rect"
        radius = float((cfg.get("robot") or {}).get("radius_m", 0.20))
        cells_x = int(round(radius / self.cell_x_m))
        cells_y = int(round(radius / self.cell_y_m))
        self.inflate_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * cells_y + 1, 2 * cells_x + 1))
            if max(cells_x, cells_y) > 0 else None)
        # Hazard pad: a ring this wide around the static chassis where a
        # predicted drop is replaced by the model's best other class.  The
        # teacher used to paint rail-edge flying pixels as a drop strip
        # along the chassis; the ring hides that on engines trained before
        # ``label.hazard_ego_margin_m`` existed and is harmless after.
        self.hazard_pad_m = float(ego.get("hazard_pad_m", 0.0) or 0.0)
        self.hazard_pad_ring = None
        px = int(round(self.hazard_pad_m / self.cell_x_m))
        py = int(round(self.hazard_pad_m / self.cell_y_m))
        if max(px, py) > 0 and self.ego_static.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * py + 1, 2 * px + 1))
            dil = cv2.dilate(self.ego_static.astype(np.uint8), k) > 0
            self.hazard_pad_ring = dil & ~self.ego_static

    @staticmethod
    def _find_meta(engine_path):
        for p in (engine_path.with_suffix(".json"),
                  engine_path.parent / "bev_student.json"):
            if p.exists():
                return p
        raise StudentRuntimeError(
            f"no metadata JSON next to {engine_path}")

    @staticmethod
    def _torch_dtype(dtype):
        import tensorrt as trt
        np_dtype = trt.nptype(dtype)
        if np_dtype == np.float16:
            return torch.float16
        if np_dtype == np.float64:
            return torch.float64
        return torch.float32

    # -- preprocessing -----------------------------------------------------

    @staticmethod
    def _color(frame, image_hw=IMG_HW):
        if frame.color_jpeg:
            bgr = cv2.imdecode(
                np.frombuffer(frame.color_jpeg, np.uint8),
                cv2.IMREAD_COLOR)
            if bgr is None:
                raise StudentRuntimeError("could not decode color JPEG")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif frame.color is not None:
            rgb = frame.color              # StampedFrame.color is RGB
        else:
            raise StudentRuntimeError("camera frame has no color")
        rgb = cv2.resize(rgb, tuple(image_hw)[::-1],
                         interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return x.transpose(2, 0, 1)

    @staticmethod
    def _depth(frame, image_hw=IMG_HW):
        d = frame.depth
        if d is None:
            raise StudentRuntimeError("camera frame has no depth")
        d = cv2.resize(d, tuple(image_hw)[::-1],
                       interpolation=cv2.INTER_NEAREST)
        scale = float(frame.depth_scale_mm or 1.0)
        m = d.astype(np.float32) * scale * 1e-3 / 8.0
        return np.clip(m, 0.0, 1.0)[None]

    def preprocess(self, frames):
        colors = []
        depths = []
        for role in ("near", "far"):
            fr = frames.get(role)
            if fr is None:
                raise StudentRuntimeError(f"missing {role} frame")
            colors.append(self._color(fr, self.image_hw))
            depths.append(self._depth(fr, self.image_hw))
        return np.concatenate(
            [np.stack(colors), np.stack(depths)], axis=1
        ).astype(np.float32)

    # -- inference ---------------------------------------------------------

    def infer(self, frames, geo_grid=None, ages_s=None):
        t0 = time.perf_counter()
        x = self.preprocess(frames)
        t_pre = time.perf_counter()
        x = torch.from_numpy(x).to(self.device)
        if x.dtype != self.input.dtype:
            x = x.to(self.input.dtype)
        self.input.copy_(x)
        self.context.execute_async_v3(self.stream)
        torch.cuda.current_stream().synchronize()
        t_infer = time.perf_counter()
        self.last_preprocess_ms = (t_pre - t0) * 1e3
        self.last_latency_ms = (t_infer - t_pre) * 1e3
        sem_logits = self.outputs["semantic"].float().cpu().numpy()
        if self.single_head:
            ego_logits = cov_logits = None
        else:
            ego_logits = self.outputs["ego"].float().cpu().numpy()
            cov_logits = self.outputs["coverage"].float().cpu().numpy()
        grid, stats = self.postprocess(
            sem_logits, ego_logits, cov_logits, frames, geo_grid, ages_s)
        stats["preprocess_ms"] = round(self.last_preprocess_ms, 2)
        stats["infer_ms"] = round(self.last_latency_ms, 2)
        self.last_latency_ms = stats["preprocess_ms"] + stats["infer_ms"]
        self.runs += 1
        return grid

    # -- postprocessing ----------------------------------------------------

    def postprocess(self, sem_logits, ego_logits, cov_logits, frames,
                    geo_grid=None, ages_s=None):
        sem_logits = np.asarray(sem_logits)
        if sem_logits.ndim == 4:
            sem_logits = sem_logits[0]
        probs = self._softmax(sem_logits, axis=0)
        semantic = probs.argmax(0).astype(np.uint8)
        if ego_logits is None or cov_logits is None:
            # single head: class 4 is the chassis, class 5 unobserved
            if sem_logits.shape[0] != 6:
                raise StudentRuntimeError(
                    "single-head engine must have 6 semantic classes")
            ego_cls = semantic == 4
            cov_cls = (semantic != 5) & ~ego_cls
            ego_logits = np.where(ego_cls, 30.0, -30.0).astype(np.float32)
            cov_logits = np.where(cov_cls, 30.0, -30.0).astype(np.float32)
        ego_logits = np.asarray(ego_logits).reshape(-1, self.rows, self.cols)[0]
        cov_logits = np.asarray(cov_logits).reshape(-1, self.rows, self.cols)[0]
        # Optional hazard probability threshold.  The default 0.5 is the
        # argmax contract; lower values trade precision for stair/drop
        # recall, higher values suppress more orange.  Applied within the
        # predicted coverage mask below.
        if getattr(self, "hazard_threshold", 0.5) != 0.5:
            hz = probs[SEM_HAZARD] > self.hazard_threshold
            if self.hazard_threshold < 0.5:
                semantic[hz] = SEM_HAZARD
            else:
                alt = probs[:3].argmax(0).astype(np.uint8)
                semantic[(semantic == SEM_HAZARD) & ~hz] = alt[
                    (semantic == SEM_HAZARD) & ~hz]
        confidence = (probs.max(0) * 255.0).astype(np.uint8)
        ego = self._sigmoid(ego_logits) > 0.5
        coverage = self._sigmoid(cov_logits) > 0.5
        ego = ego | self.ego_static

        model_hazard = semantic == SEM_HAZARD
        geo_hazard = None
        if geo_grid is not None:
            geo_hazard = (np.asarray(geo_grid.raw) == HAZARD)
        if self.hazard_source == "geometric" and geo_hazard is not None:
            # Replace the (unreliable) student hazard with the geometric
            # hazard source.  The best non-hazard class keeps floor/wall/
            # obstacle coherent where the model guessed hazard.
            alt = probs[:3].argmax(0).astype(np.uint8)
            semantic = np.where(model_hazard, alt, semantic).astype(np.uint8)
            semantic[geo_hazard] = SEM_HAZARD
            coverage = coverage | geo_hazard
        elif self.hazard_source == "union":
            if geo_hazard is not None:
                semantic[geo_hazard] = SEM_HAZARD
                coverage = coverage | geo_hazard
        # "model" leaves the student hazard untouched.
        ring = getattr(self, "hazard_pad_ring", None)
        edge_hazard = 0
        if ring is not None:
            edge = ring & (semantic == SEM_HAZARD)
            edge_hazard = int(edge.sum())
            if edge_hazard:
                alt = probs[:3].argmax(0).astype(np.uint8)
                semantic = semantic.copy()
                semantic[edge] = alt[edge]

        coverage = coverage & ~ego
        semantic = semantic.copy()
        semantic[~coverage] = 255
        semantic[ego] = 255
        wall = (semantic == SEM_WALL)
        raw = np.zeros((self.rows, self.cols), np.uint8)
        raw[(semantic == SEM_WALL) | (semantic == SEM_OBSTACLE)] = OCCUPIED
        raw[semantic == SEM_HAZARD] = HAZARD
        raw[~coverage] = FREE
        raw[ego] = FREE
        occ = raw.copy()
        if self.inflate_kernel is not None:
            blocked = ((raw == OCCUPIED) | (raw == HAZARD)).astype(np.uint8)
            dil = cv2.dilate(blocked, self.inflate_kernel)
            occ[(dil > 0) & (occ == FREE)] = INFLATED
        source = np.zeros((self.rows, self.cols), np.uint8)
        source[coverage] = SRC_BOTH
        # Keep source useful even if a camera was missing; both normally.
        ages = dict(ages_s or {})
        stamp = max(int(fr.stamp_us) for fr in frames.values() if fr)
        stats = {
            "engine": Path(self.engine_path).name,
            "hazard_source": self.hazard_source,
            "semantic_counts": {
                "floor": int(np.sum(semantic == SEM_FLOOR)),
                "wall": int(np.sum(semantic == SEM_WALL)),
                "obstacle": int(np.sum(semantic == SEM_OBSTACLE)),
                "hazard": int(np.sum(semantic == SEM_HAZARD)),
                "unknown": int(np.sum(semantic == 255)),
            },
            "coverage_cells": int(np.sum(coverage)),
            "ego_cells": int(np.sum(ego)),
            "edge_hazard_cells": edge_hazard,
            "roles": sorted(frames),
        }
        grid = BevGrid(
            occ=occ, raw=raw, coverage=coverage, source=source,
            confidence=confidence, cell_m=self.cell_m,
            cell_x_m=self.cell_x_m, cell_y_m=self.cell_y_m,
            range_m=self.range_m, width_m=self.width_m,
            stamp_us=stamp, recv_ts=time.monotonic(),
            per_camera_age_s=ages, plane_ok={}, plane={},
            contributions={"model": stats}, roles=sorted(frames),
            semantic=semantic, wall=wall, ego=ego)
        return grid, stats

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    def _softmax(x, axis=0):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def status(self):
        return {
            "engine": Path(self.engine_path).name,
            "hazard_source": self.hazard_source,
            "hazard_threshold": getattr(self, "hazard_threshold", 0.5),
            "input_shape": list(self.in_shape),
            "infer_ms": (round(self.last_latency_ms, 2)
                         if self.last_latency_ms is not None else None),
            "runs": self.runs,
        }
