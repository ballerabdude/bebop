"""navd telemetry messages, Protobuf-encoded (`bebop.navd.*`).

The recorder writes `/cmd_vel` and `/odom` as these messages so Rerun's
generic Protobuf MCAP decoder exposes their fields as a struct component
(`bebop.navd.Twist:message`), which the dashboard's `SeriesLines`
visualizers map to time-series via field selectors — no export step needed.

Sources: `bebop-vision/protos/navd/telemetry.proto`, compiled with:

    protoc -I protos --python_out=bebop_vision/proto navd/telemetry.proto
"""

from bebop_vision.foxglove_proto import file_descriptor_set
from bebop_vision.proto.navd import telemetry_pb2

Twist = telemetry_pb2.Twist
Odom = telemetry_pb2.Odom
ImuAccel = telemetry_pb2.ImuAccel
ImuGyro = telemetry_pb2.ImuGyro


def schema(message_cls):
    """(name, schema_encoding, data) for MCAP register_schema."""
    return (message_cls.DESCRIPTOR.full_name, "protobuf",
            file_descriptor_set(message_cls))
