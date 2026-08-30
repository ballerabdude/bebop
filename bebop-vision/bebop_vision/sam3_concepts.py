"""SAM 3 / SAM 3.1 text-prompted concept segmentation (Meta sam3 package).

License note: SAM 3.x code and weights are under Meta's "SAM License" (NOT Apache).
Commercial use and fine-tuning are permitted and you own your derivative works,
but redistribution of SAM 3.x materials or derivatives must carry the SAM License.
This module is teacher-side tooling (dataset recording / labeling); the robot's
runtime uses the distilled SegFormer nav model only.
"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from . import config
from .runtime import autocast_ctx
from .segmenter import InstanceSegment

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

SAM3_CKPTS = {
    "sam3": os.path.join(WEIGHTS_DIR, "sam3.pt"),
    "sam3.1": os.path.join(WEIGHTS_DIR, "sam3.1_multiplex.pt"),
}


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.numpy()
    return np.asarray(x)


class TRTEncoder:
    """TensorRT-backed SAM 3.1 vision encoder (drop-in for backbone.forward_image)."""

    OUT_NAMES = [
        "vision_features",
        "vision_pos_0", "vision_pos_1", "vision_pos_2",
        "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    ]

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_buf = torch.empty((1, 3, 1008, 1008), device=device, dtype=torch.float32)
        self.out_bufs = {
            name: torch.empty(tuple(self.engine.get_tensor_shape(name)), device=device, dtype=torch.float32)
            for name in self.OUT_NAMES
        }
        self.context.set_tensor_address("image", self.input_buf.data_ptr())
        for name, buf in self.out_bufs.items():
            self.context.set_tensor_address(name, buf.data_ptr())
        self._stream = torch.cuda.current_stream().cuda_stream

    def __call__(self, image):
        import tensorrt as trt  # noqa: F401  (keep enum import local)

        self.input_buf.copy_(image)
        self.context.execute_async_v3(self._stream)
        o = self.out_bufs
        return {
            "vision_features": o["vision_features"],
            "vision_pos_enc": [o["vision_pos_0"], o["vision_pos_1"], o["vision_pos_2"]],
            "backbone_fpn": [o["backbone_fpn_0"], o["backbone_fpn_1"], o["backbone_fpn_2"]],
            "sam2_backbone_out": None,
        }


class Sam3ConceptSegmenter:
    def __init__(self, concepts=None, conf=0.3, device=config.DEVICE, version="sam3.1", resolution=1008,
                 trt_engine=None):
        if resolution != 1008:
            raise ValueError(
                "SAM 3 image mode only supports resolution=1008 "
                "(Meta package hardcodes RoPE/position encoding at 1008)"
            )
        if version not in SAM3_CKPTS:
            raise ValueError(f"SAM version must be one of {list(SAM3_CKPTS)}")
        ckpt = SAM3_CKPTS[version]
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt} not found. Request access at https://huggingface.co/facebook/{version}, "
                "then run: python -m bebop_vision.download_sam3"
            )
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.concepts = [c.lower().strip() for c in concepts] if concepts else list(config.RECORD_CONCEPTS)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        builder_device = "cuda" if device.startswith("cuda") else "cpu"
        self.model = build_sam3_image_model(checkpoint_path=ckpt, device=builder_device)
        self.processor = Sam3Processor(self.model, resolution=resolution, device=builder_device)
        if trt_engine is not None:
            self.model.backbone.forward_image = TRTEncoder(trt_engine, device=builder_device)
            print(f"[sam3] TRT encoder active: {trt_engine}")
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        self.segment(dummy)
        print(f"[sam3] ready ({time.perf_counter() - start:.2f}s warmup, concepts: {', '.join(self.concepts)})")

    def segment(self, frame):
        """Find and mask all instances of the configured concepts in a BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        segments = []
        with torch.inference_mode(), autocast_ctx(self.device):
            state = self.processor.set_image(Image.fromarray(rgb))
            for class_id, concept in enumerate(self.concepts):
                output = self.processor.set_text_prompt(state=state, prompt=concept)
                boxes = _to_numpy(output.get("boxes"))
                scores = _to_numpy(output.get("scores"))
                masks = _to_numpy(output.get("masks"))
                if boxes is None or boxes.size == 0:
                    continue
                for i in range(boxes.shape[0]):
                    score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                    if score < self.conf:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in boxes[i][:4])
                    mask = None
                    if masks is not None and i < len(masks):
                        m = np.squeeze(np.asarray(masks[i]))
                        if m.ndim == 2:
                            mask = m.astype(bool)
                    segments.append(
                        InstanceSegment(
                            bbox=(x1, y1, x2, y2),
                            confidence=score,
                            class_id=class_id,
                            label=concept,
                            mask=mask,
                        )
                    )
        return segments