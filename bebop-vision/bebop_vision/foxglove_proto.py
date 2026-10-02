"""Foxglove well-known messages, Protobuf-encoded (foxglove-sdk schemas).

The recorder writes these as MCAP protobuf schemas instead of JSON so that
Rerun's `foxglove` MCAP decoder (which is Protobuf-only) maps image/video
topics to Rerun archetypes. Foxglove reads protobuf-encoded Foxglove
messages too, so a single encoding serves both viewers.

The `.proto` sources live in `bebop-vision/protos/foxglove/` and are
compiled with:

    protoc -I protos -I /usr/include --python_out=bebop_vision/proto \\
        foxglove/CompressedImage.proto foxglove/CompressedVideo.proto \\
        foxglove/RawImage.proto
"""

from google.protobuf import descriptor_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from bebop_vision.proto.foxglove import CompressedImage_pb2
from bebop_vision.proto.foxglove import CompressedVideo_pb2
from bebop_vision.proto.foxglove import RawImage_pb2

CompressedImage = CompressedImage_pb2.CompressedImage
CompressedVideo = CompressedVideo_pb2.CompressedVideo
RawImage = RawImage_pb2.RawImage


def timestamp(log_ns):
    """google.protobuf.Timestamp from nanoseconds since the epoch."""
    ts = Timestamp()
    ts.seconds = int(log_ns // 1_000_000_000)
    ts.nanos = int(log_ns % 1_000_000_000)
    return ts


def file_descriptor_set(message_cls):
    """Serialized FileDescriptorSet for a message's file and dependencies.

    MCAP's schema data for a protobuf schema is a serialized
    `google.protobuf.FileDescriptorSet`; Rerun/Foxglove build the message
    descriptor from it to decode payloads.
    """
    order = []
    seen = set()

    def visit(fd):
        if fd.name in seen:
            return
        seen.add(fd.name)
        for dep in fd.dependencies:
            visit(dep)
        order.append(fd)

    visit(message_cls.DESCRIPTOR.file)
    fds = descriptor_pb2.FileDescriptorSet()
    for fd in order:
        proto = descriptor_pb2.FileDescriptorProto()
        fd.CopyToProto(proto)
        fds.file.append(proto)
    return fds.SerializeToString()


def schema(message_cls):
    """(name, schema_encoding, data) for MCAP register_schema."""
    return (message_cls.DESCRIPTOR.full_name, "protobuf",
            file_descriptor_set(message_cls))
