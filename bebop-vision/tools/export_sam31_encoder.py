import sys
import time

sys.path.insert(0, "/mnt/data/projects/bebop/bebop-vision")

import torch

import sam3.model.vitdet as vitdet
import sam3.model_builder as mb
from sam3.model_builder import build_sam3_image_model


def addmm_act_fp16(activation, linear, mat1):
    y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return torch.relu(y)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return torch.nn.functional.gelu(y)
    raise ValueError


vitdet.addmm_act = addmm_act_fp16
_orig_vit = mb._create_vit_backbone
mb._create_vit_backbone = lambda *a, **k: _orig_vit(*a, **{**k, "use_rope_real": True})

OUT = "/mnt/data/projects/bebop/bebop-vision/weights/sam31_encoder_fp16.onnx"


class EncWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        x = image.to(torch.float16)
        out = self.backbone.forward_image(x)
        return (
            out["vision_features"].float(),
            out["vision_pos_enc"][0].float(), out["vision_pos_enc"][1].float(),
            out["vision_pos_enc"][2].float(),
            out["backbone_fpn"][0].float(), out["backbone_fpn"][1].float(),
            out["backbone_fpn"][2].float(),
        )


m = build_sam3_image_model(
    checkpoint_path="/mnt/data/projects/bebop/bebop-vision/weights/sam3.1_multiplex.pt",
    device="cuda", load_from_HF=False,
)
for p in m.parameters():
    if p.is_floating_point():
        p.data = p.data.to(torch.float16)

w = EncWrapper(m.backbone).eval().cuda()
dummy = torch.randn(1, 3, 1008, 1008, device="cuda", dtype=torch.float32)

t0 = time.perf_counter()
with torch.inference_mode():
    torch.onnx.export(
        w, (dummy,), OUT,
        input_names=["image"],
        output_names=[
            "vision_features",
            "vision_pos_0", "vision_pos_1", "vision_pos_2",
            "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
        ],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
print(f"native-fp16 ONNX export done in {time.perf_counter() - t0:.0f}s")

import onnx

model = onnx.load(OUT)
onnx.checker.check_model(model)
print("ONNX check ok |", len(model.graph.node), "nodes")
