//! Embedded ROS2 .msg definitions as const strings.
//!
//! These are embedded in the MCAP schema registry with
//! `ros2msg` encoding so Foxglove Studio's ROS2 panel can
//! decode every channel without external configuration.
//!
//! Multi-section schemas (with dependencies) use the standard
//! `====` separator between `.msg` file sections.

/// sensor_msgs/msg/JointState — positions + velocities for 8 joints.
pub const JOINT_STATE_SCHEMA_NAME: &str = "sensor_msgs/msg/JointState";

pub const JOINT_STATE_SCHEMA: &str = "\
# Standard ROS2 sensor_msgs/JointState\n\
# =======================================\n\
std_msgs/Header header\n\
string[] name\n\
float64[] position\n\
float64[] velocity\n\
float64[] effort\n\
====\n\
MSG: std_msgs/Header\n\
builtin_interfaces/Time stamp\n\
string frame_id\n\
====\n\
MSG: builtin_interfaces/Time\n\
int32 sec\n\
uint32 nanosec\n";

/// sensor_msgs/msg/Imu — body-frame orientation + angular velocity.
pub const IMU_SCHEMA_NAME: &str = "sensor_msgs/msg/Imu";

pub const IMU_SCHEMA: &str = "\
# Standard ROS2 sensor_msgs/Imu\n\
# =============================\n\
std_msgs/Header header\n\
geometry_msgs/Quaternion orientation\n\
float64[9] orientation_covariance\n\
geometry_msgs/Vector3 angular_velocity\n\
float64[9] angular_velocity_covariance\n\
geometry_msgs/Vector3 linear_acceleration\n\
float64[9] linear_acceleration_covariance\n\
====\n\
MSG: std_msgs/Header\n\
builtin_interfaces/Time stamp\n\
string frame_id\n\
====\n\
MSG: builtin_interfaces/Time\n\
int32 sec\n\
uint32 nanosec\n\
====\n\
MSG: geometry_msgs/Quaternion\n\
float64 x 0\n\
float64 y 0\n\
float64 z 0\n\
float64 w 1\n\
====\n\
MSG: geometry_msgs/Vector3\n\
float64 x 0\n\
float64 y 0\n\
float64 z 0\n";

/// bebop_msgs/msg/PolicyStatus — metadata for each tick.
pub const POLICY_STATUS_SCHEMA_NAME: &str = "bebop_msgs/msg/PolicyStatus";

pub const POLICY_STATUS_SCHEMA: &str = "\
# Bebop policy status per tick\n\
# =============================\n\
std_msgs/Header header\n\
string mode\n\
bool dry_run\n\
bool imu_live\n\
float64 sim_time_s\n\
====\n\
MSG: std_msgs/Header\n\
builtin_interfaces/Time stamp\n\
string frame_id\n\
====\n\
MSG: builtin_interfaces/Time\n\
int32 sec\n\
uint32 nanosec\n";

/// bebop_msgs/msg/Float32Stamped — timestamped float array.
pub const OBSERVATION_SCHEMA_NAME: &str = "bebop_msgs/msg/Float32Stamped";

pub const OBSERVATION_SCHEMA: &str = "\
# Timestamped float32 array\n\
# =========================\n\
std_msgs/Header header\n\
float32[] data\n\
====\n\
MSG: std_msgs/Header\n\
builtin_interfaces/Time stamp\n\
string frame_id\n\
====\n\
MSG: builtin_interfaces/Time\n\
int32 sec\n\
uint32 nanosec\n";

/// bebop_msgs/msg/PolicyAction — NN output + decoded targets.
pub const POLICY_ACTION_SCHEMA_NAME: &str = "bebop_msgs/msg/PolicyAction";

pub const POLICY_ACTION_SCHEMA: &str = "\
# Bebop policy action output\n\
# ===========================\n\
std_msgs/Header header\n\
float32[] raw_action\n\
float32[] position_targets_rad\n\
float32[] kp\n\
float32[] kd\n\
====\n\
MSG: std_msgs/Header\n\
builtin_interfaces/Time stamp\n\
string frame_id\n\
====\n\
MSG: builtin_interfaces/Time\n\
int32 sec\n\
uint32 nanosec\n";
