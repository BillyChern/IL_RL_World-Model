# ARX X5 SDK Exploration

**Repository:** https://github.com/ARXroboticsX/ARX_X5
**Contact:** contact@arx-x.com
**License:** BSD-3-Clause
**Last Updated:** 2025-10-18

## Table of Contents
1. [SDK Overview](#sdk-overview)
2. [Architecture](#architecture)
3. [Python SDK](#python-sdk)
4. [Control Interfaces](#control-interfaces)
5. [Proprioception & Sensing](#proprioception--sensing)
6. [Action Space Specification](#action-space-specification)
7. [ROS Integration](#ros-integration)
8. [Dependencies & Setup](#dependencies--setup)
9. [Example Usage](#example-usage)
10. [Safety Features & Limitations](#safety-features--limitations)
11. [Integration Recommendations](#integration-recommendations)

---

## SDK Overview

The ARX X5 SDK is a comprehensive robotics control framework supporting multiple middleware platforms and control paradigms. It provides:

- **Multi-framework support**: ROS, ROS2, standalone C++, and Python
- **Real-time control**: CAN bus communication for low-latency robot control
- **Flexible interfaces**: Joint-space and Cartesian-space control modes
- **Dual-arm capabilities**: Native support for bimanual manipulation
- **VR integration**: Teleoperation support through ARX_VR_SDK

### Repository Structure

```
ARX_X5/
├── 00-sh/                    # Shell utility scripts
├── ARX_CAN/                  # CAN bus communication layer
│   ├── arx_can/             # Core CAN implementation
│   ├── arx_can.rules        # udev rules for CAN devices
│   └── *.sh                 # CAN setup and configuration scripts
├── ARX_VR_SDK/              # Virtual reality teleoperation
├── ROS/X5_ws/               # ROS (1) workspace
│   └── src/
│       ├── ARX_X5_ros_V7/arx_x5_controller/  # Main ROS controller
│       └── arx_msg_ros/     # Custom ROS message definitions
├── ROS2/X5_ws/              # ROS2 workspace
├── arx_joy/                 # Joystick/gamepad control
├── py/arx_x5_python/        # Python SDK (primary focus)
│   ├── bimanual/            # Single and dual-arm Python interface
│   ├── test_single_arm.py   # Single arm example
│   ├── test_dual_arm.py     # Dual arm example
│   └── test_keyboard.py     # Keyboard teleoperation
└── 旧版-readme/             # Legacy documentation (Chinese)
```

### Key Components

1. **ARX_CAN**: Low-level CAN bus communication providing hardware interface
2. **Python SDK**: High-level API with `SingleArm` and `BimanualArm` classes
3. **ROS Controller**: Full-featured ROS node with multiple control modes
4. **VR SDK**: Teleoperation for human demonstrations and interventions

---

## Architecture

### Communication Stack

```
┌─────────────────────────────────┐
│   Application Layer             │
│  (Python/ROS/C++ User Code)     │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Control Layer                 │
│  - SingleArm / BimanualArm      │
│  - X5Controller (ROS)           │
│  - Inverse Kinematics           │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   CAN Communication Layer       │
│  - arx_can library              │
│  - Message formatting           │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Hardware                      │
│  - CAN bus interface            │
│  - ARX X5 Robot Arm             │
└─────────────────────────────────┘
```

### Multi-Framework Integration

The SDK supports three primary integration approaches:

1. **Python Direct** (Recommended for RL/IL): Lightweight, minimal dependencies, direct CAN control
2. **ROS/ROS2**: Full middleware stack, topic-based communication, ecosystem integration
3. **C++ Direct**: Maximum performance, embedded systems, custom control loops

---

## Python SDK

### Installation & Setup

#### Dependencies

```bash
# System dependencies
sudo apt update
sudo apt install can-utils

# Python dependencies
sudo pip3 install keyboard numpy

# pybind11 (for building from source)
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake ..
make
sudo make install
```

#### Building the SDK

```bash
cd ARX_X5/py/arx_x5_python/
./build.sh
source ./setup.sh
```

#### CAN Interface Configuration

```bash
# Configure CAN device (required before each session)
sudo slcand -o -f -s8 /dev/arxcan0 can0
sudo ifconfig can0 up

# Verify CAN interface
ifconfig can0
```

### SingleArm Class

The `SingleArm` class provides a unified interface for controlling a single 6-DOF robotic arm with gripper.

#### Constructor

```python
from bimanual import SingleArm
import numpy as np

# Configuration options
config = {
    "num_joints": 7,           # 6 arm joints + 1 gripper (default: 7)
    "dt": 0.05,                # Control timestep in seconds (default: 0.05 = 20Hz)
    "type": 0,                 # URDF model: 0=x5, 1=x5_master, 2=x5_2025
    "can_port": "can0"         # CAN interface name
}

arm = SingleArm(config)
```

#### Control Methods

**System Commands**
```python
# Move to predefined home position
success = arm.go_home()  # Returns: bool

# Enable gravity compensation (zero-gravity mode)
success = arm.gravity_compensation()  # Returns: bool

# Activate protective/safe mode
success = arm.protect_mode()  # Returns: bool
```

**Joint-Space Control**
```python
# Command joint positions (6 joints)
positions = [0.0, 0.5, -0.3, 0.0, 0.8, 0.0]  # radians
success = arm.set_joint_positions(positions)  # Returns: bool

# Single joint control
success = arm.set_joint_positions(0.5)  # Command first joint only
```

**Cartesian-Space Control**
```python
# End-effector control with quaternion orientation
position = np.array([0.3, 0.0, 0.2])  # [x, y, z] in meters
quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
success = arm.set_ee_pose(position, quaternion)  # Returns: bool

# End-effector control with Euler angles (RPY)
xyzrpy = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0])  # [x, y, z, roll, pitch, yaw]
success = arm.set_ee_pose_xyzrpy(xyzrpy)  # Returns: bool
```

**Gripper Control**
```python
# Set gripper position
arm.set_catch_pos(0.0)   # Fully open
arm.set_catch_pos(1.0)   # Fully closed
arm.set_catch_pos(0.5)   # Half-closed
```

#### Proprioception Methods

**Joint State**
```python
# Get current joint angles (6 DOF)
joint_positions = arm.get_joint_positions()  # Returns: ndarray, shape (6,)

# Get joint velocities
joint_velocities = arm.get_joint_velocities()  # Returns: ndarray, shape (6,)

# Get joint motor currents (proxy for torques)
joint_currents = arm.get_joint_currents()  # Returns: ndarray, shape (6,)
```

**End-Effector State**
```python
# Get end-effector pose with quaternion
pos, quat = arm.get_ee_pose()
# pos: ndarray, shape (3,) - [x, y, z] in meters
# quat: ndarray, shape (4,) - [w, x, y, z]

# Get end-effector pose with Euler angles
xyzrpy = arm.get_ee_pose_xyzrpy()  # Returns: ndarray, shape (6,)
```

### BimanualArm Class

Manages dual-arm coordination by wrapping two `SingleArm` instances.

#### Constructor

```python
from bimanual import BimanualArm

# Separate configurations for each arm
left_config = {
    "can_port": "can1",
    "type": 0,
    "dt": 0.05
}

right_config = {
    "can_port": "can0",
    "type": 0,
    "dt": 0.05
}

dual_arm = BimanualArm(left_config, right_config)
```

#### Coordinate System

**Important:** The BimanualArm uses a centered coordinate frame:
- **Origin**: Center of shoulder (between two arms)
- **X-axis**: Forward (positive = away from robot)
- **Y-axis**: Right (positive = right side)
- **Z-axis**: Down (positive = downward)

#### Control Methods

**System Commands**
```python
# Return both arms to home position
status = dual_arm.go_home()
# Returns: {"left": bool, "right": bool}

# Enable gravity compensation on both arms
status = dual_arm.gravity_compensation()
# Returns: {"left": bool, "right": bool}
```

**Joint-Space Control**
```python
# Get joint names for querying/setting
joint_names = dual_arm.get_joint_names("left")   # Left arm joints
joint_names = dual_arm.get_joint_names("right")  # Right arm joints
joint_names = dual_arm.get_joint_names("both")   # All joints

# Set specific joints by name
positions = {
    "left_joint_1": 0.5,
    "right_joint_1": -0.5
}
joint_names = ["left_joint_1", "right_joint_1"]
dual_arm.set_joint_positions(positions, joint_names)

# Get joint positions
positions = dual_arm.get_joint_positions("left", joint_names)
# Returns: dict with joint_name -> angle mapping

# Get joint velocities
velocities = dual_arm.get_joint_velocities("right", joint_names)
```

**Cartesian-Space Control**
```python
# Set poses for both arms simultaneously
poses = {
    'left': (
        np.array([0.2, 0.1, 0.15]),      # position [x, y, z]
        np.array([1.0, 0.0, 0.0, 0.0])   # quaternion [w, x, y, z]
    ),
    'right': (
        np.array([0.2, -0.1, 0.15]),
        np.array([1.0, 0.0, 0.0, 0.0])
    )
}
dual_arm.set_ee_pose(poses)

# Alternative: RPY-based control
xyzrpy = {
    'left': np.array([0.2, 0.1, 0.15, 0.0, 0.0, 0.0]),
    'right': np.array([0.2, -0.1, 0.15, 0.0, 0.0, 0.0])
}
dual_arm.set_ee_pose_rpy(xyzrpy)

# Get end-effector poses
left_pos, left_quat = dual_arm.get_ee_pose("left")
right_pos, right_quat = dual_arm.get_ee_pose("right")
```

---

## Control Interfaces

### Control Frequency

The SDK supports variable control rates depending on the interface:

| Interface | Frequency | Configuration | Notes |
|-----------|-----------|---------------|-------|
| Python SDK | 20 Hz (default) | `dt=0.05` in config | Configurable via `dt` parameter |
| Python SDK | Up to 50 Hz | `dt=0.02` in config | Tested in examples |
| ROS Controller | 100 Hz | Hardcoded in X5Controller.cpp | State publishing rate |
| CAN Bus | Hardware dependent | System-level | Underlying communication layer |

**For RL/IL applications:** The default 20 Hz (dt=0.05) is suitable, but 50 Hz can be achieved by setting `dt=0.02` for finer control granularity.

### Control Modes

The ARX X5 supports multiple control paradigms:

#### 1. Position Control (Primary)

**Joint Position Control**
- Direct specification of joint angles
- 6-DOF arm control
- Suitable for trajectory following

**Cartesian Position Control**
- End-effector pose commands
- Automatic inverse kinematics
- Choice of quaternion or Euler angle representation

#### 2. Gravity Compensation Mode

Enables manual manipulation with gravity forces compensated. Useful for:
- Human demonstrations
- Kinesthetic teaching
- Safety testing

#### 3. Protective Mode

Low-level safety mode for:
- Emergency stops
- Collision avoidance
- Force limiting (implementation details not documented)

### ROS Control Interface

The `X5Controller` ROS node provides a more feature-rich interface with:

**Multiple Command Formats**
- **V1 Format**: `[x, y, z, roll, pitch, yaw, gripper]` via `PosCmd` message
- **V2 Format**: `[end_pos[6], joint_pos[6], mode, gripper]` via `RobotCmd` message
- **Joint Control**: Direct joint position commands

**Control Loop**
- Main loop: 100 Hz state publishing
- Command processing: Callback-based
- Inverse kinematics: Integrated for Cartesian commands

**ROS Topics** (configurable in launch file)
- Input: `/arm_cmd` (default command topic)
- Input: `/arx_joy` (joystick for homing/compensation)
- Output: `/arm_status_ee` (end-effector pose)
- Output: `/arm_status_joint` (joint states)
- Output: `/arm_status` (combined status)

---

## Proprioception & Sensing

### Available Sensor Data

The ARX X5 provides comprehensive proprioceptive feedback:

| Data Type | Python Method | ROS Topic | Dimensions | Units |
|-----------|---------------|-----------|------------|-------|
| Joint Angles | `get_joint_positions()` | `/arm_status_joint` | (6,) | radians |
| Joint Velocities | `get_joint_velocities()` | `/arm_status_joint` | (6,) | rad/s |
| Joint Currents | `get_joint_currents()` | `/arm_status_joint` | (6,) | amperes |
| EE Position | `get_ee_pose()[0]` | `/arm_status_ee` | (3,) | meters |
| EE Orientation (quat) | `get_ee_pose()[1]` | `/arm_status_ee` | (4,) | [w,x,y,z] |
| EE Orientation (RPY) | `get_ee_pose_xyzrpy()[3:6]` | `/arm_status_ee` | (3,) | radians |
| Gripper Position | Not explicitly documented | - | (1,) | normalized [0,1] |

### Torque Estimation

**Note:** True torque sensors are not explicitly documented. Joint currents serve as a proxy:
- Motor current is proportional to torque
- Use `get_joint_currents()` for force feedback
- Calibration may be required for accurate torque values

### Camera Integration

**Status:** No camera integration code found in the SDK.

The repository search found no references to:
- Camera drivers
- Image topics
- Vision sensor configuration
- RGB or depth camera support

**For RL/IL integration:**
- External camera setup required
- ROS `usb_cam` or `realsense2_camera` can be used separately
- Synchronization between camera and robot state must be handled externally

---

## Action Space Specification

### Recommended Action Space for RL/IL

For imitation learning and reinforcement learning, we recommend the following action space design:

#### Option 1: Joint Position Control (Recommended)

**Dimensionality:** 7D (6 arm joints + 1 gripper)

```python
action = np.array([
    θ1,  # Joint 1 angle (radians)
    θ2,  # Joint 2 angle (radians)
    θ3,  # Joint 3 angle (radians)
    θ4,  # Joint 4 angle (radians)
    θ5,  # Joint 5 angle (radians)
    θ6,  # Joint 6 angle (radians)
    g    # Gripper position [0.0=open, 1.0=closed]
])

# Execution
arm.set_joint_positions(action[:6])
arm.set_catch_pos(action[6])
```

**Advantages:**
- Direct hardware control (no IK solver delay)
- Reproducible trajectories
- Lower latency
- Well-defined joint space

**Disadvantages:**
- Less intuitive for task specification
- Requires learning inverse kinematics implicitly

#### Option 2: Cartesian Position Control (Delta Actions)

**Dimensionality:** 7D (3 position + 3 rotation + 1 gripper)

```python
# Current state
current_xyzrpy = arm.get_ee_pose_xyzrpy()

# Delta action (small increments)
delta_action = np.array([
    Δx,      # Position change in X (meters, e.g., ±0.01)
    Δy,      # Position change in Y (meters)
    Δz,      # Position change in Z (meters)
    Δroll,   # Rotation change in roll (radians, e.g., ±0.05)
    Δpitch,  # Rotation change in pitch (radians)
    Δyaw,    # Rotation change in yaw (radians)
    g        # Gripper position [0.0, 1.0]
])

# New target pose
target_xyzrpy = current_xyzrpy + delta_action[:6]

# Execution
arm.set_ee_pose_xyzrpy(target_xyzrpy)
arm.set_catch_pos(delta_action[6])
```

**Advantages:**
- Task-space control more intuitive
- Easier transfer between robots
- Better for Cartesian-space tasks

**Disadvantages:**
- IK solver may fail near singularities
- Slightly higher latency

#### Option 3: Hybrid (Position + Velocity)

Combine position control with velocity feedback for more responsive policies:

```python
# Action includes both target and velocity hints
action = {
    "position": θ_target,   # Target joint positions
    "velocity": θ_dot_desired  # Desired joint velocities (for smoother control)
}
```

This requires lower-level CAN access and is more complex but offers better tracking.

### Action Space Bounds

**Critical:** No explicit joint limits are documented in the SDK.

**Recommendations:**
1. **Empirical Testing**: Manually move robot through full range to determine limits
2. **Conservative Bounds**: Start with restricted range (e.g., ±π/2) and expand gradually
3. **Software Clipping**: Implement action clipping in your RL environment:

```python
# Example conservative bounds (VERIFY THESE ON HARDWARE)
JOINT_LIMITS_LOWER = np.array([-2.0, -1.5, -2.0, -1.5, -2.0, -1.5])  # radians
JOINT_LIMITS_UPPER = np.array([2.0, 1.5, 2.0, 1.5, 2.0, 1.5])         # radians

def clip_action(action):
    action[:6] = np.clip(action[:6], JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    action[6] = np.clip(action[6], 0.0, 1.0)  # Gripper
    return action
```

---

## ROS Integration

### ROS 1 Controller (X5Controller)

The ROS controller provides a full-featured interface with:

**Features:**
- 100 Hz state publishing
- Multiple command message types (V1, V2, joint)
- Automatic inverse kinematics
- Joystick integration for homing/gravity compensation
- Configurable topic names and parameters

**Key Implementation Details** (from X5Controller.cpp):

```cpp
// Control frequency
ros::Duration(0.01)  // 100 Hz timer

// Proprioception access
interfaces_ptr_->getJointPositons()   // 7-DOF joint angles
interfaces_ptr_->getJointVelocities() // 7-DOF joint velocities
interfaces_ptr_->getJointCurrent()    // 7-DOF motor currents
interfaces_ptr_->getEndPose()         // End-effector Isometry3d

// Control modes
- End-effector Cartesian control (with IK)
- Direct joint position control
- Gripper position control
- Gripper torque control (selectable via CatchControlMode param)
```

**Configuration** (arm_config.yaml):

```yaml
remote_master_l:
  go_home_position: [-0.112, 0.948, 0.858, -0.573, -0.105, 0.097]

remote_master_r:
  go_home_position: [-0.112, 0.948, 0.858, -0.573, -0.105, 0.097]
```

### ROS 2 Support

The repository includes a ROS2 workspace (`ROS2/X5_ws/`), but detailed documentation is not available. The structure suggests similar functionality to ROS 1.

### Custom ROS Messages

The `arx_msg_ros` package provides:
- Custom message types for ARX X5 communication
- Standardized interface across ROS nodes
- Integration with standard ROS message types

---

## Dependencies & Setup

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04/22.04 recommended)
- Real-time kernel optional but recommended for low-latency control

**Hardware:**
- USB-to-CAN adapter (or built-in CAN interface)
- ARX X5 robot arm(s)
- Sufficient CPU for control loop (modest requirements, ~1 core)

### Software Dependencies

**Core Dependencies:**
```bash
# CAN utilities
sudo apt install can-utils

# Python libraries
sudo pip3 install numpy keyboard

# C++ build tools (if building from source)
sudo apt install build-essential cmake git

# pybind11 (for Python bindings)
git clone https://github.com/pybind/pybind11.git
cd pybind11 && mkdir build && cd build
cmake .. && make && sudo make install
```

**ROS Dependencies** (if using ROS interface):
```bash
# ROS Noetic (Ubuntu 20.04) or ROS Humble (Ubuntu 22.04)
# Follow official ROS installation instructions

# Additional ROS packages
sudo apt install ros-$ROS_DISTRO-ros-control
sudo apt install ros-$ROS_DISTRO-ros-controllers
```

### Installation Steps

#### Python SDK Installation

```bash
# 1. Clone repository
git clone https://github.com/ARXroboticsX/ARX_X5.git
cd ARX_X5/py/arx_x5_python/

# 2. Build Python bindings
./build.sh

# 3. Setup environment
source ./setup.sh

# 4. Configure CAN interface (before each session)
sudo slcand -o -f -s8 /dev/arxcan0 can0
sudo ifconfig can0 up

# 5. Test installation
python3 test_single_arm.py
```

#### ROS Installation

```bash
# 1. Clone repository into ROS workspace
cd ~/catkin_ws/src  # or ~/ros2_ws/src for ROS2
git clone https://github.com/ARXroboticsX/ARX_X5.git

# 2. Copy relevant packages
cp -r ARX_X5/ROS/X5_ws/src/* .

# 3. Build workspace
cd ~/catkin_ws
catkin_make  # or colcon build for ROS2

# 4. Source workspace
source devel/setup.bash

# 5. Launch controller
roslaunch arx_x5_controller x5_controller.launch
```

### CAN Interface Configuration

**One-time setup** (udev rules for persistent device names):

```bash
# Copy udev rules
sudo cp ARX_X5/ARX_CAN/arx_can.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Per-session setup:**

```bash
# Option 1: Using slcand (software CAN daemon)
sudo slcand -o -f -s8 /dev/arxcan0 can0
sudo ifconfig can0 up

# Option 2: Using ip command (for hardware CAN)
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# Verify
ifconfig can0
candump can0  # Should show CAN traffic when robot is active
```

---

## Example Usage

### Example 1: Basic Joint Control (Python)

```python
#!/usr/bin/env python3
import numpy as np
import time
from bimanual import SingleArm

# Initialize robot
config = {
    "can_port": "can0",
    "type": 0,
    "dt": 0.02  # 50 Hz control
}
arm = SingleArm(config)

# Move to home position
print("Moving to home...")
arm.go_home()
time.sleep(2.0)

# Joint control loop
print("Starting joint control loop...")
for i in range(100):
    # Sinusoidal motion on joint 2
    t = i * config["dt"]
    joint_positions = [0.0, 0.5 * np.sin(t), 0.0, 0.0, 0.0, 0.0]

    arm.set_joint_positions(joint_positions)

    # Read proprioception
    current_positions = arm.get_joint_positions()
    current_velocities = arm.get_joint_velocities()

    print(f"Step {i}: Joint 2 target={joint_positions[1]:.3f}, "
          f"actual={current_positions[1]:.3f}, "
          f"velocity={current_velocities[1]:.3f}")

    time.sleep(config["dt"])

print("Complete!")
```

### Example 2: Cartesian Control with Gripper (Python)

```python
#!/usr/bin/env python3
import numpy as np
import time
from bimanual import SingleArm

# Initialize
config = {"can_port": "can0", "type": 0, "dt": 0.05}
arm = SingleArm(config)

# Go home
arm.go_home()
time.sleep(2.0)

# Circular motion in XY plane
print("Circular trajectory...")
center = np.array([0.3, 0.0, 0.2])  # [x, y, z] in meters
radius = 0.05
duration = 5.0  # seconds
steps = int(duration / config["dt"])

for i in range(steps):
    t = i * config["dt"]
    angle = 2 * np.pi * t / duration

    # Circular position
    position = center + np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        0.0
    ])

    # Fixed orientation (pointing down)
    orientation = np.array([0.0, 0.0, 0.0])  # RPY: [roll, pitch, yaw]
    xyzrpy = np.concatenate([position, orientation])

    # Gripper opens/closes sinusoidally
    gripper_pos = 0.5 + 0.5 * np.sin(2 * angle)

    # Send commands
    arm.set_ee_pose_xyzrpy(xyzrpy)
    arm.set_catch_pos(gripper_pos)

    # Feedback
    if i % 10 == 0:
        ee_pos, ee_quat = arm.get_ee_pose()
        print(f"t={t:.2f}s: Target={position}, Actual={ee_pos}, Gripper={gripper_pos:.2f}")

    time.sleep(config["dt"])

# Close gripper and return home
arm.set_catch_pos(1.0)
time.sleep(1.0)
arm.go_home()
```

### Example 3: Bimanual Coordination (Python)

```python
#!/usr/bin/env python3
import numpy as np
import time
from bimanual import BimanualArm

# Initialize dual arms
left_config = {"can_port": "can1", "type": 0, "dt": 0.05}
right_config = {"can_port": "can0", "type": 0, "dt": 0.05}
dual_arm = BimanualArm(left_config, right_config)

# Go home
dual_arm.go_home()
time.sleep(2.0)

# Symmetric reaching motion
print("Bimanual reaching...")
for i in range(100):
    t = i * 0.05

    # Both arms move forward and outward
    x = 0.25 + 0.05 * np.sin(t)
    y_offset = 0.15 + 0.05 * np.cos(t)
    z = 0.2

    poses = {
        'left': (
            np.array([x, y_offset, z]),       # Position
            np.array([1.0, 0.0, 0.0, 0.0])    # Orientation (quaternion)
        ),
        'right': (
            np.array([x, -y_offset, z]),      # Mirror position
            np.array([1.0, 0.0, 0.0, 0.0])
        )
    }

    dual_arm.set_ee_pose(poses)

    if i % 20 == 0:
        left_pos, _ = dual_arm.get_ee_pose("left")
        right_pos, _ = dual_arm.get_ee_pose("right")
        print(f"Step {i}: Left={left_pos}, Right={right_pos}")

    time.sleep(0.05)

print("Returning home...")
dual_arm.go_home()
```

### Example 4: RL Environment Integration (Pseudocode)

```python
import gym
from gym import spaces
import numpy as np
from bimanual import SingleArm

class ARXX5Env(gym.Env):
    """Custom Gym environment for ARX X5 robot."""

    def __init__(self, task="reach", camera_ids=[0], control_hz=20):
        super().__init__()

        # Initialize robot
        self.arm = SingleArm({
            "can_port": "can0",
            "type": 0,
            "dt": 1.0 / control_hz
        })

        # Define action space: 7D (6 joints + gripper)
        self.action_space = spaces.Box(
            low=np.array([-2.0, -1.5, -2.0, -1.5, -2.0, -1.5, 0.0]),
            high=np.array([2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 1.0]),
            dtype=np.float32
        )

        # Define observation space
        # State: joint pos (6) + joint vel (6) + ee pos (3) + ee quat (4) + gripper (1) = 20
        # Image: 224x224x3 RGB
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32),
            "image": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
        })

        # Camera setup (external, not part of SDK)
        self.camera_ids = camera_ids
        self.cameras = self._init_cameras()

        self.task = task
        self.control_dt = 1.0 / control_hz

    def reset(self):
        """Reset environment to initial state."""
        # Go to home position
        self.arm.go_home()
        time.sleep(1.0)

        # Randomize object placement (task-specific)
        self._reset_objects()

        # Get observation
        obs = self._get_obs()
        return obs

    def step(self, action):
        """Execute action and return observation, reward, done, info."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Send commands
        self.arm.set_joint_positions(action[:6])
        self.arm.set_catch_pos(action[6])

        # Wait for control cycle
        time.sleep(self.control_dt)

        # Get observation
        obs = self._get_obs()

        # Compute reward (task-specific)
        reward = self._compute_reward(obs, action)

        # Check termination
        done = self._check_done(obs)

        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        """Get current observation."""
        # Robot state
        joint_pos = self.arm.get_joint_positions()
        joint_vel = self.arm.get_joint_velocities()
        ee_pos, ee_quat = self.arm.get_ee_pose()
        gripper_pos = 0.0  # Gripper state not directly available

        state = np.concatenate([
            joint_pos,      # (6,)
            joint_vel,      # (6,)
            ee_pos,         # (3,)
            ee_quat,        # (4,)
            [gripper_pos]   # (1,)
        ])

        # Camera image (external camera, not part of SDK)
        image = self._get_camera_image()

        return {"state": state, "image": image}

    def _get_camera_image(self):
        """Capture camera image (external implementation)."""
        # This requires separate camera driver (e.g., OpenCV, ROS usb_cam)
        # Not part of ARX X5 SDK
        import cv2
        cap = cv2.VideoCapture(self.camera_ids[0])
        ret, frame = cap.read()
        cap.release()
        return cv2.resize(frame, (224, 224))

    def _compute_reward(self, obs, action):
        """Compute reward (task-specific)."""
        if self.task == "reach":
            # Example: distance to target
            ee_pos = obs["state"][12:15]
            target_pos = self.target_pos
            distance = np.linalg.norm(ee_pos - target_pos)
            reward = -distance
            return reward
        else:
            raise NotImplementedError

    def _check_done(self, obs):
        """Check if episode is complete."""
        # Task-specific termination conditions
        return False

    def _reset_objects(self):
        """Reset task objects (external manipulation)."""
        # Place objects randomly on table
        # This is task-specific and requires external setup
        pass

    def _init_cameras(self):
        """Initialize cameras (external)."""
        # Setup cameras (RealSense, USB, etc.)
        return []
```

---

## Safety Features & Limitations

### Documented Safety Features

1. **Protective Mode**
   - `arm.protect_mode()` activates a safety state
   - Specific behavior not documented (likely torque/velocity limiting)

2. **Gravity Compensation**
   - `arm.gravity_compensation()` allows safe manual manipulation
   - Robot becomes compliant for kinesthetic teaching

3. **Home Position**
   - Predefined safe configuration
   - Can be used for emergency recovery

### Critical Limitations

**WARNING: The following safety features are NOT found in the SDK documentation:**

1. **No Explicit Joint Limits**
   - No documented position limits for joints
   - Software bounds must be implemented by user
   - Risk of mechanical damage if limits exceeded

2. **No Documented Velocity Limits**
   - Maximum safe joint velocities not specified
   - Rapid motions could cause instability or damage

3. **No Torque/Force Limits**
   - Joint current readings available but no force limits documented
   - Collision detection must be implemented externally

4. **No Collision Avoidance**
   - No self-collision checking
   - No environment collision detection
   - Dual-arm systems risk collision between arms

5. **No E-Stop Integration**
   - No documented emergency stop mechanism in software
   - Physical e-stop button recommended but not documented

### Recommended Safety Measures

For RL/IL applications, implement the following safety layers:

#### 1. Action Clipping

```python
# Define conservative joint limits (VERIFY ON HARDWARE)
JOINT_LIMITS_LOWER = np.array([-2.0, -1.5, -2.0, -1.5, -2.0, -1.5])
JOINT_LIMITS_UPPER = np.array([2.0, 1.5, 2.0, 1.5, 2.0, 1.5])

def safe_action(action):
    """Clip action to safe range."""
    return np.clip(action, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
```

#### 2. Velocity Monitoring

```python
MAX_JOINT_VELOCITY = 1.0  # rad/s (conservative, TUNE FOR YOUR SETUP)

def check_velocity(arm):
    """Check if joint velocities are within safe range."""
    velocities = arm.get_joint_velocities()
    if np.any(np.abs(velocities) > MAX_JOINT_VELOCITY):
        print("WARNING: Excessive joint velocity detected!")
        arm.protect_mode()
        return False
    return True
```

#### 3. Watchdog Timer

```python
import time

class RobotWatchdog:
    """Watchdog to detect control loop failures."""

    def __init__(self, timeout=0.5):
        self.timeout = timeout
        self.last_update = time.time()

    def update(self):
        """Call this in every control loop iteration."""
        self.last_update = time.time()

    def check(self, arm):
        """Check if control loop is responsive."""
        if time.time() - self.last_update > self.timeout:
            print("ERROR: Control loop timeout!")
            arm.protect_mode()
            return False
        return True
```

#### 4. Workspace Limits

```python
def check_workspace(ee_pos):
    """Verify end-effector is within safe workspace."""
    # Define safe Cartesian workspace
    X_RANGE = [0.1, 0.5]  # meters
    Y_RANGE = [-0.3, 0.3]
    Z_RANGE = [0.0, 0.4]

    if not (X_RANGE[0] <= ee_pos[0] <= X_RANGE[1]):
        print("WARNING: X position out of workspace!")
        return False
    if not (Y_RANGE[0] <= ee_pos[1] <= Y_RANGE[1]):
        print("WARNING: Y position out of workspace!")
        return False
    if not (Z_RANGE[0] <= ee_pos[2] <= Z_RANGE[1]):
        print("WARNING: Z position out of workspace!")
        return False

    return True
```

#### 5. Human-in-the-Loop Safety

For RL training (especially HIL-SERL approach):

```python
class HumanInterventionMonitor:
    """Monitor for human takeovers during RL training."""

    def __init__(self, arm):
        self.arm = arm
        self.intervention_active = False

    def check_intervention(self):
        """Check if human has taken control (e.g., via force sensing)."""
        # Monitor joint currents for external forces
        currents = self.arm.get_joint_currents()
        threshold = 0.5  # amperes, TUNE THIS

        if np.any(np.abs(currents) > threshold):
            print("Human intervention detected!")
            self.intervention_active = True
            self.arm.gravity_compensation()  # Allow manual control
            return True

        return False

    def resume_autonomous(self):
        """Resume autonomous control after intervention."""
        # Wait for human to release
        time.sleep(1.0)
        self.intervention_active = False
        print("Resuming autonomous control...")
```

### Physical Safety Setup

**Strongly recommended:**

1. **Emergency Stop Button**: Hardware e-stop within reach
2. **Workspace Barriers**: Physical guards to prevent collisions
3. **Soft Materials**: Foam padding on robot and obstacles during training
4. **Human Supervision**: Always have operator present during RL training
5. **Gradual Deployment**: Test in simulation first, then constrained environments

---

## Integration Recommendations

### For IL/RL World Model Project

Based on the CLAUDE.md project specifications, here are specific recommendations:

#### 1. Control Interface Selection

**Recommendation: Use Python SDK with Joint Position Control**

**Rationale:**
- Direct CAN communication minimizes latency
- 20-50 Hz control suitable for manipulation tasks
- Simpler than ROS for RL training loops
- Full proprioception access

**Implementation:**
```python
# In your RL environment
from bimanual import SingleArm

class ARXEnv:
    def __init__(self):
        self.robot = SingleArm({
            "can_port": "can0",
            "type": 0,
            "dt": 0.02  # 50 Hz
        })
```

#### 2. Observation Space Design

**Recommended observation:**
```python
observation = {
    # Proprioception (20D)
    "robot_state": np.concatenate([
        arm.get_joint_positions(),   # (6,)
        arm.get_joint_velocities(),  # (6,)
        ee_pos,                      # (3,) from get_ee_pose()
        ee_quat,                     # (4,)
        [gripper_pos]                # (1,) estimated
    ]),

    # Vision (external cameras, NOT in SDK)
    "camera_0": rgb_image,  # (224, 224, 3)
    "camera_1": rgb_image,  # Optional: multiple views

    # Language instruction (for VLA)
    "instruction": "pick up the red block"
}
```

#### 3. Action Space Design

**Recommended: 7D Joint Position Control**
```python
action_space = gym.spaces.Box(
    low=np.array([-2.0, -1.5, -2.0, -1.5, -2.0, -1.5, 0.0]),
    high=np.array([2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 1.0]),
    dtype=np.float32
)
```

**Alternative: 7D Delta Actions (for finer control)**
```python
# Smaller increments for smoother learning
DELTA_MAX = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1])  # radians and gripper

def apply_delta_action(current_state, delta_action):
    new_action = current_state + delta_action
    return safe_clip(new_action)
```

#### 4. Camera Setup (External)

**The ARX X5 SDK does NOT include camera support.**

**Recommended external setup:**

```python
import cv2
import pyrealsense2 as rs

class CameraManager:
    """Manage external cameras for visual observations."""

    def __init__(self, camera_type="realsense"):
        if camera_type == "realsense":
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
        elif camera_type == "usb":
            self.cap = cv2.VideoCapture(0)

    def get_image(self):
        """Capture RGB image."""
        if hasattr(self, 'pipeline'):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            return np.asanyarray(color_frame.get_data())
        else:
            ret, frame = self.cap.read()
            return frame
```

**Camera placement recommendations:**
- **Wrist-mounted**: Track end-effector and gripper (requires external mounting)
- **Third-person fixed**: Workspace overview for spatial reasoning
- **Dual cameras**: Stereo depth perception

#### 5. Reward Function

**For learned reward classifier (as per CLAUDE.md):**

```python
class LearnedRewardModel:
    """Binary reward classifier from demonstrations."""

    def __init__(self):
        # Train on successful vs. failed demonstrations
        self.model = self._train_classifier()

    def _train_classifier(self):
        """Train binary classifier on demo data."""
        # Load demos: (observation, action, success_label)
        # Train neural network or ensemble
        pass

    def compute_reward(self, observation, action):
        """Compute reward for current state-action pair."""
        # Success probability from classifier
        success_prob = self.model.predict(observation, action)
        return success_prob
```

**Combined with task-specific metrics:**
```python
def compute_reward(obs, action, task):
    """Hybrid reward: learned + task-specific."""

    # Learned component
    learned_reward = reward_model.compute_reward(obs, action)

    # Task-specific component (e.g., distance to target)
    if task == "reach":
        ee_pos = obs["robot_state"][12:15]
        target_pos = task_params["target_pos"]
        distance_reward = -np.linalg.norm(ee_pos - target_pos)

    # Weighted combination
    total_reward = 0.7 * learned_reward + 0.3 * distance_reward
    return total_reward
```

#### 6. Human-in-the-Loop Integration (HIL-SERL)

**For human interventions during RL training:**

```python
class HILCollector:
    """Collect data with human interventions."""

    def __init__(self, arm, intervention_detector):
        self.arm = arm
        self.detector = intervention_detector
        self.intervention_buffer = []

    def collect_episode(self, policy):
        """Run episode with possible human takeover."""
        obs = self.reset()
        done = False
        trajectory = []

        while not done:
            # Check for human intervention
            if self.detector.check_intervention():
                # Human is controlling - record as high-value demonstration
                self.arm.gravity_compensation()

                while self.detector.intervention_active:
                    # Record human actions
                    obs_next = self._get_obs()
                    action_human = self._infer_action(obs, obs_next)
                    trajectory.append((obs, action_human, obs_next, 1.0, "human"))
                    obs = obs_next
                    time.sleep(0.05)

                self.detector.resume_autonomous()
            else:
                # Autonomous policy control
                action = policy.predict(obs)
                obs_next, reward, done, info = self.step(action)
                trajectory.append((obs, action, obs_next, reward, "policy"))
                obs = obs_next

        # Store with intervention annotations
        self._store_trajectory(trajectory)
        return trajectory

    def _infer_action(self, obs_prev, obs_curr):
        """Infer human's action from state change."""
        # Compute joint position differences
        joint_prev = obs_prev["robot_state"][:6]
        joint_curr = obs_curr["robot_state"][:6]
        action_inferred = joint_curr - joint_prev
        return action_inferred
```

#### 7. World Model Integration

**For DayDreamer-style imagined rollouts:**

```python
class WorldModel:
    """Learned dynamics model for imagined experience."""

    def __init__(self, obs_dim, action_dim):
        # Latent dynamics model (Dreamer-style)
        # or video prediction model (JEPA/Diffusion)
        self.encoder = ...
        self.dynamics = ...
        self.decoder = ...
        self.reward_model = ...

    def train(self, replay_buffer):
        """Update world model on real experience."""
        batch = replay_buffer.sample()

        # Encode observations to latent space
        latent = self.encoder(batch["obs"])

        # Predict next latent state
        latent_next_pred = self.dynamics(latent, batch["action"])
        latent_next_true = self.encoder(batch["obs_next"])

        # Reconstruction loss
        loss_dynamics = F.mse_loss(latent_next_pred, latent_next_true)

        # Reward prediction loss
        reward_pred = self.reward_model(latent, batch["action"])
        loss_reward = F.mse_loss(reward_pred, batch["reward"])

        # Train
        total_loss = loss_dynamics + loss_reward
        total_loss.backward()
        self.optimizer.step()

    def imagine_trajectory(self, start_obs, policy, horizon=10):
        """Generate imagined trajectory."""
        latent = self.encoder(start_obs)
        imagined_trajectory = []

        for t in range(horizon):
            # Policy action in latent space
            action = policy.predict(latent)

            # Predict next latent state
            latent_next = self.dynamics(latent, action)

            # Predict reward
            reward = self.reward_model(latent, action)

            imagined_trajectory.append((latent, action, reward))
            latent = latent_next

        return imagined_trajectory
```

#### 8. Asynchronous Training Loop

**Parallel real robot collection + GPU training:**

```python
import threading
import queue

class AsynchronousTrainer:
    """Asynchronous RL training with real robot and world model."""

    def __init__(self, robot_env, policy, world_model, replay_buffer):
        self.robot_env = robot_env
        self.policy = policy
        self.world_model = world_model
        self.replay_buffer = replay_buffer

        self.data_queue = queue.Queue(maxsize=100)
        self.running = True

    def robot_collection_thread(self):
        """Collect real robot experience (runs on CPU)."""
        while self.running:
            # Collect episode
            trajectory = self.robot_env.collect_episode(self.policy)

            # Add to replay buffer
            for transition in trajectory:
                self.replay_buffer.add(transition)
                self.data_queue.put(transition)

            print(f"Collected episode: {len(trajectory)} steps")

    def training_thread(self):
        """Train on GPU while robot collects."""
        while self.running:
            # Update world model on latest data
            if len(self.replay_buffer) > 1000:
                self.world_model.train(self.replay_buffer)

            # Generate imagined rollouts
            batch_real = self.replay_buffer.sample(batch_size=32)
            imagined_trajectories = []
            for obs in batch_real["obs"]:
                imag_traj = self.world_model.imagine_trajectory(obs, self.policy, horizon=10)
                imagined_trajectories.append(imag_traj)

            # Train policy on mixture of real + imagined
            batch_combined = self._combine_real_imagined(batch_real, imagined_trajectories)
            policy_loss = self.policy.train(batch_combined)

            print(f"Policy loss: {policy_loss:.4f}")

            time.sleep(0.1)  # Adjust based on compute

    def train(self, num_episodes):
        """Run asynchronous training."""
        # Start threads
        robot_thread = threading.Thread(target=self.robot_collection_thread)
        training_thread = threading.Thread(target=self.training_thread)

        robot_thread.start()
        training_thread.start()

        # Monitor training
        try:
            robot_thread.join()
            training_thread.join()
        except KeyboardInterrupt:
            print("Stopping training...")
            self.running = False
```

#### 9. VLA Policy Integration

**For language-conditioned policy (π₀.₅-style):**

```python
class VLAPolicy:
    """Vision-Language-Action policy for manipulation."""

    def __init__(self, vision_encoder, language_encoder):
        # Pre-trained encoders (e.g., CLIP, DINOv2, BERT)
        self.vision_encoder = vision_encoder  # Frozen or lightly fine-tuned
        self.language_encoder = language_encoder  # Frozen

        # Action decoder (trainable)
        self.action_decoder = MLPActionDecoder(input_dim=512+768, output_dim=7)

    def predict(self, observation, instruction):
        """Predict action from multimodal observation."""
        # Encode vision
        image = observation["camera_0"]
        vision_features = self.vision_encoder(image)  # (512,)

        # Encode language
        language_features = self.language_encoder(instruction)  # (768,)

        # Concatenate with robot state
        robot_state = observation["robot_state"]  # (20,)

        # Fuse features
        multimodal_features = torch.cat([
            vision_features,
            language_features,
            torch.tensor(robot_state)
        ], dim=-1)

        # Predict action
        action = self.action_decoder(multimodal_features)  # (7,)
        return action.detach().cpu().numpy()
```

#### 10. Deployment Checklist

Before running RL training on ARX X5:

- [ ] Test manual control with `test_single_arm.py`
- [ ] Verify CAN communication stability
- [ ] Empirically determine joint limits (manual testing)
- [ ] Implement software safety layers (action clipping, velocity monitoring)
- [ ] Setup external cameras and test synchronization
- [ ] Collect initial demonstration dataset (10-50 episodes)
- [ ] Pre-train VLA policy on demonstrations
- [ ] Pre-train world model on demonstrations
- [ ] Test world model prediction accuracy (MSE < threshold)
- [ ] Configure human intervention protocol (e-stop, gravity comp button)
- [ ] Setup safety barriers and soft padding
- [ ] Run short RL trials with human supervision (10-20 episodes)
- [ ] Monitor for hardware issues (overheating, unusual sounds)
- [ ] Gradually increase autonomy as policy improves

---

## Summary

The ARX X5 SDK provides a flexible, multi-framework control system suitable for imitation learning and reinforcement learning research. Key takeaways:

**Strengths:**
- Clean Python API with both single and dual-arm support
- Real-time CAN communication for low-latency control
- Multiple control modes (joint, Cartesian, gravity compensation)
- Comprehensive proprioception (joint angles, velocities, currents)
- ROS integration available for ecosystem compatibility

**Limitations:**
- **No camera integration** (requires external setup)
- **No documented safety limits** (joint limits, velocity limits, torque limits)
- **Limited documentation** (no explicit API reference or hardware specs)
- **No collision avoidance** (user must implement)
- **No force/torque sensors** (only motor currents available)

**Critical Recommendations for RL/IL:**
1. Implement comprehensive software safety layers before deployment
2. Use external camera system synchronized with robot control
3. Empirically determine joint limits through manual testing
4. Start with conservative action bounds and expand gradually
5. Always maintain human supervision during training
6. Pre-train on demonstrations before RL fine-tuning
7. Use joint position control (7D action space) for direct hardware access
8. Implement asynchronous data collection for sample efficiency

**Best Use Cases:**
- Research in vision-language-action models
- Human-in-the-loop reinforcement learning
- World model-based RL with parallel imagination
- Bimanual manipulation tasks
- Rapid prototyping of manipulation policies

This SDK aligns well with the HIL-SERL + DayDreamer + VLA architecture described in CLAUDE.md, providing the necessary low-level control while allowing flexibility for high-level learning algorithms.

---

## Additional Resources

- **Repository:** https://github.com/ARXroboticsX/ARX_X5
- **Contact:** contact@arx-x.com
- **License:** BSD-3-Clause
- **Related Projects:**
  - π₀.₅ (pi0.5): https://github.com/Physical-Intelligence/pi0
  - HIL-SERL: https://github.com/rail-berkeley/serl
  - DayDreamer: https://github.com/danijar/daydreamer
  - DreamerV3: https://github.com/danijar/dreamerv3

**Disclaimer:** This documentation is based on analysis of the ARX X5 SDK GitHub repository as of 2025-10-18. Always verify behavior on actual hardware and consult with ARXrobotics for official specifications and safety guidelines.
