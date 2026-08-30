import tensorrt as trt

ONNX = "/mnt/data/projects/bebop/bebop-vision/weights/sam31_encoder_fp16.onnx"
ENGINE = "/mnt/data/projects/bebop/bebop-vision/weights/sam31_encoder_fp16.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

with open(ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(min(10, parser.num_errors)):
            print("parse error:", parser.get_error(i))
        raise SystemExit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

serialized = builder.build_serialized_network(network, config)
assert serialized is not None, "engine build failed"
with open(ENGINE, "wb") as f:
    f.write(serialized)
print(f"engine built -> {ENGINE}")
