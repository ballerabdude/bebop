#!/bin/bash
# set -e

# Source the ROS 2 setup script
source /opt/ros/jazzy/setup.bash

# Define workspace path (set in Dockerfile)
ROS_WS=${ROS_WS:-"/home/$USER/master_ros2_ws"}

echo "=== Bebop ROS 2 Workspace ==="

# Function to setup micro-ROS (only if not present)
setup_micro_ros() {
    local SRC_DIR="$ROS_WS/src"
    
    # Clone micro_ros_setup if needed
    if [ ! -d "$SRC_DIR/micro_ros_setup" ]; then
        echo "→ Cloning micro_ros_setup..."
        cd "$SRC_DIR"
        git clone -b jazzy https://github.com/micro-ROS/micro_ros_setup.git
        
        # Install rosdeps
        cd "$ROS_WS"
        sudo rosdep update 2>/dev/null || true
        sudo rosdep install --from-paths src --ignore-src -y 2>/dev/null || true
        
        # Build micro_ros_setup
        echo "→ Building micro_ros_setup..."
        colcon build --packages-select micro_ros_setup --symlink-install
        source "$ROS_WS/install/setup.bash"
    fi
    
    # Create and build micro-ROS agent if needed
    if [ ! -d "$SRC_DIR/uros" ]; then
        echo "→ Creating micro-ROS agent workspace..."
        source "$ROS_WS/install/setup.bash"
        ros2 run micro_ros_setup create_agent_ws.sh
        
        echo "→ Building micro-ROS agent (this takes ~2 minutes)..."
        ros2 run micro_ros_setup build_agent.sh
        echo "✓ micro-ROS agent ready!"
    fi
}

# Function to build workspace if needed
build_workspace() {
    cd "$ROS_WS"
    if [ ! -d "$ROS_WS/install" ]; then
        echo "→ Building workspace..."
        colcon build --symlink-install
    fi
}

# ============================================
# Main Setup
# ============================================

cd "$ROS_WS"

# Setup micro-ROS if not present
if [ ! -d "$ROS_WS/src/uros/micro-ROS-Agent" ]; then
    echo "First run: Setting up micro-ROS..."
    setup_micro_ros
fi

# Build workspace if needed
build_workspace

# Source the workspace
if [ -f "$ROS_WS/install/setup.bash" ]; then
    source "$ROS_WS/install/setup.bash"
fi

echo "✓ Workspace ready: $ROS_WS"
echo ""

# Execute command passed to entrypoint
exec "$@"
