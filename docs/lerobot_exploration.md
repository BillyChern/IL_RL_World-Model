# LeRobot Repository Exploration

## Overview

LeRobot is Hugging Face's official library for end-to-end robot learning, designed to democratize AI for robotics by providing standardized datasets, pretrained models, and training infrastructure. It implements state-of-the-art approaches including imitation learning (ACT policy), diffusion-based policies, and reinforcement learning (TDMPC).

**Repository**: https://github.com/huggingface/lerobot
**License**: Apache 2.0
**Documentation**: https://huggingface.co/docs/lerobot

---

## RLDS Format Specification

### What is RLDS?

RLDS (Reinforcement Learning Datasets) is Google Research's ecosystem for storing, retrieving, and manipulating episodic data for Sequential Decision Making, Reinforcement Learning, Learning from Demonstrations, Offline RL, and Imitation Learning.

**Repository**: https://github.com/google-research/rlds

### RLDS Data Structure

RLDS organizes data hierarchically as nested TensorFlow datasets:

#### Episode Level Structure

Episodes are dictionaries containing:
- A `tf.data.Dataset` of Steps
- User-defined metadata fields

**Recommended Episode Metadata**:
```python
{
    "episode_id": <unique_identifier>,      # Unique across datasets for merging
    "agent_id": <agent_identifier>,         # Supports multi-agent scenarios
    "environment_config": <env_setup>,      # Environment configuration details
    "experiment_id": <experiment_id>,       # Experiment identifier
    "invalid": <bool>,                      # Flag for incomplete/filtered episodes
    "steps": <tf.data.Dataset>              # Nested dataset of steps
}
```

#### Step Level Structure

Steps are dictionaries with mandatory and optional fields:

**Mandatory Fields**:
```python
{
    "is_first": <bool>,    # True for episode start with initial state
    "is_last": <bool>      # True for episode end (invalidates action/reward/discount)
}
```

**Optional Fields**:
```python
{
    "observation": <state>,      # Current state observation
    "action": <action>,          # Applied action
    "reward": <float>,           # Resulting reward
    "discount": <float>,         # Discount factor
    "is_terminal": <bool>,       # Terminal state indicator
    # ... additional custom metadata
}
```

### RLDS Data Flow Convention

Following dm_env conventions, data generation flows as:
```
(o₀, _, _, _, m₀) → (o₁, a₀, r₀, d₀, m₁) → (o₂, a₁, r₁, d₁, m₂)
```

When loaded via RLDS, observations align with corresponding actions:
```
(o₀, a₀, r₀, d₀, m₀) → (o₁, a₁, r₁, d₁, m₁) → (o₂, a₂, r₂, d₂, m₂)
```

### Creating RLDS Datasets

**Using EnvLogger for Synthetic Data**:
```python
import envlogger

env = envlogger.EnvLogger(
    environment,
    data_directory='/tmp/mydataset'
)
```

EnvLogger wraps dm_env environments and records agent-environment interactions with optional per-step and per-episode metadata callbacks.

---

## LeRobotDataset Format

### Format Overview

LeRobotDataset is a standardized format that addresses robotics-specific needs by providing unified access to sensorimotor data, multiple camera feeds, and teleoperation metadata. It combines three storage methods:

1. **Parquet files** (via Hugging Face datasets) - Structured tabular data
2. **MP4 video files** - Visual observations
3. **JSON/JSONL files** - Metadata and configuration

### Dataset Structure (v3.0)

#### Directory Organization

```
dataset_root/
├── meta/
│   ├── info.json              # Dataset schema, features, shapes, fps
│   ├── stats.json             # Normalization statistics (mean, std, min, max)
│   ├── tasks.jsonl            # Task descriptions mapped to integer indices
│   └── episodes/              # Chunked Parquet files with episode metadata
│       ├── chunk_0000.parquet
│       └── chunk_0001.parquet
├── data/                      # Concatenated tabular data
│   ├── chunk_0000.parquet
│   └── chunk_0001.parquet
└── videos/                    # MP4 files organized by camera and chunk
    ├── camera_0/
    │   ├── chunk_0000.mp4
    │   └── chunk_0001.mp4
    └── camera_1/
        ├── chunk_0000.mp4
        └── chunk_0001.mp4
```

#### Standard Data Fields

**Tabular Data (Parquet files)**:
```python
{
    "observation.state": <proprioceptive_data>,    # Joint angles, end-effector pos
    "action": <commands>,                          # Target joint angles/velocities
    "timestamp": <seconds_from_start>,             # Seconds from episode start
    "episode_index": <int>,                        # Episode identifier
    "frame_index": <int>,                          # Frame position in episode
    "index": <int>,                                # Global unique frame ID
    "next.done": <bool>,                           # Episode termination flag
    "task_index": <int>                            # Optional task identifier
}
```

**Visual Data (Video files)**:
```python
{
    "observation.images.<camera_name>": {
        "path": <video_file_path>,
        "timestamp": <frame_timing>
    }
}
```

**Metadata Files**:
- `info.json`: Dataset-wide configuration (codebase_version, robot_type, fps, feature schemas)
- `tasks.jsonl`: Task ID-to-description mappings for language-conditioned policies
- `episodes.jsonl`: Per-episode metadata (episode_index, tasks, length)
- `stats.json` (v2.1+): Per-episode statistics for normalization

### Key Design Features

#### Delta Timestamps (Temporal Context)

LeRobotDataset supports temporal window queries via `delta_timestamps`, enabling algorithms to access observation history and action sequences:

```python
delta_timestamps = {
    # Load 4 images: 1s, 500ms, 200ms before current, and current frame
    "observation.images.front": [-1.0, -0.5, -0.2, 0.0],

    # Load 6 state vectors at different time offsets
    "observation.state": [-1.5, -1.0, -0.5, -0.2, -0.1, 0.0],

    # Load 64 action vectors: current frame and 63 future frames
    "action": [t / dataset.fps for t in range(64)]
}

dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
sample = dataset[0]

# Resulting shapes:
# sample["observation.images.front"].shape -> (4, C, H, W)
# sample["observation.state"].shape -> (6, state_dim)
# sample["action"].shape -> (64, action_dim)
```

#### Action Timing Convention

**Important**: The action recorded at frame `t` represents the command that produced the observation at frame `t+1`, not the current observation.

#### Multi-Episode File Consolidation

Unlike one-episode-per-file approaches, LeRobotDataset v3.0 consolidates multiple episodes into single files with relational metadata for episode-level retrieval. This addresses filesystem limitations when scaling to millions of episodes.

---

## Data Loading and Usage

### Basic Loading

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load from Hugging Face Hub (cached locally)
dataset = LeRobotDataset("lerobot/aloha_static_coffee")

# Load from local directory
dataset = LeRobotDataset("path/to/local/dataset", local_files_only=True)

# Access single frame
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'observation.images.top', 'action', 'timestamp', ...])
```

### Complete Usage Example

```python
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
repo_id = "lerobot/aloha_static_coffee"
dataset = LeRobotDataset(repo_id)

# Access sample
sample = dataset[100]
print(sample)
# {
#   'observation.state': tensor([...]),              # Proprioceptive state
#   'action': tensor([...]),                         # Action command
#   'observation.images.top': tensor([C, H, W]),    # Camera image
#   'observation.images.wrist': tensor([C, H, W]),  # Wrist camera
#   'timestamp': tensor(1.234),                      # Timing
#   'episode_index': tensor(5),                      # Episode ID
#   'frame_index': tensor(12),                       # Frame in episode
#   'task_index': tensor(0)                          # Task identifier
# }
```

### Using Delta Timestamps for Temporal Context

```python
# Define temporal windows
delta_timestamps = {
    "observation.images.top": [-0.2, -0.1, 0.0],    # 3 frames: 200ms, 100ms ago, current
    "observation.state": [-0.1, 0.0],                # 2 states: 100ms ago, current
    "action": [0.0, 0.1, 0.2]                        # 3 actions: current, 100ms, 200ms ahead
}

dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
sample = dataset[100]

# Temporal dimensions added
print(sample["observation.images.top"].shape)  # [3, C, H, W] - 3 time steps
print(sample["observation.state"].shape)       # [2, state_dim]
print(sample["action"].shape)                  # [3, action_dim]
```

### PyTorch DataLoader Integration

```python
from torch.utils.data import DataLoader

# Create DataLoader
batch_size = 16
num_workers = 4
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"

for batch in dataloader:
    # Move to device
    observations = batch["observation.state"].to(device)
    actions = batch["action"].to(device)
    images = batch["observation.images.top"].to(device)

    # Forward pass
    # predictions = model(observations, images)
    # loss = criterion(predictions, actions)
    pass
```

### Accessing Language Instructions (Tasks)

```python
# Load dataset
dataset = LeRobotDataset("lerobot/aloha_static_coffee")

# Access task information
sample = dataset[0]
task_index = sample["task_index"].item()

# Task descriptions stored in metadata
# Located in: meta/tasks.jsonl
# Format: {"task_index": 0, "task": "Pick the black cube and place in bowl"}

# When recording new data with specific task:
# Use command: --dataset.single_task="Grab the black cube"
```

---

## Creating Custom Datasets

### Dataset Creation API

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Define dataset features
features = {
    "observation.images.front": {
        "dtype": "video",
        "shape": (3, 480, 640),  # (C, H, W)
        "names": ["channel", "height", "width"],
        "video_info": {"fps": 30, "codec": "av1"}
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),  # 7 joint angles
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    }
}

# Create dataset
dataset = LeRobotDataset.create(
    repo_id="your_username/my_robot_dataset",
    fps=30,
    robot_type="panda",  # Must match existing types or add custom
    features=features,
    use_videos=True
)

# Add episodes
for episode_idx in range(num_episodes):
    # Collect episode data
    episode_data = collect_episode()  # Your data collection function

    # Add to dataset
    for step_idx, step_data in enumerate(episode_data):
        dataset.add_frame({
            "observation.images.front": step_data["image"],
            "observation.state": step_data["joint_angles"],
            "action": step_data["action"],
            "timestamp": step_idx / fps,
            "episode_index": episode_idx,
            "frame_index": step_idx,
            "next.done": step_idx == len(episode_data) - 1
        })

    dataset.save_episode(episode_idx)

# Consolidate and push to Hub
dataset.consolidate()
dataset.push_to_hub()
```

### Recording Real Robot Data

LeRobot provides tools for recording data directly from robots:

```bash
# Record data from robot
python lerobot/scripts/record.py \
    --robot=koch \
    --fps=30 \
    --repo-id=your_username/my_dataset \
    --dataset.single_task="Pick and place the cube" \
    --num-episodes=50

# The dataset is automatically:
# 1. Saved in LeRobotDataset format
# 2. Pushed to Hugging Face Hub (if authenticated)
```

### Dataset Naming Conventions

**Feature Naming**: Use format `<modality>.<location>`

Examples:
- `observation.images.top`
- `observation.images.front`
- `observation.images.wrist.left`
- `observation.state`
- `observation.effort`

**Avoid device-specific names** like "laptop_camera" or "phone_camera".

**Robot Types**: Select from official LeRobot config registry or ensure consistency with existing datasets.

### Quality Requirements

**Image Quality**:
- Minimum 2 camera perspectives
- Steady, shake-free video at ~30 FPS
- Consistent lighting and sharp focus
- Minimum 480x640 resolution (720p preferred)
- Exclude leader arm and human limbs from frame

**Task Descriptions**:
- Clear, concise (25-50 characters)
- Specific actions: "Pick the yellow lego block and put it in the box"
- Avoid generic labels: "task1", "demo2"

---

## RLDS to LeRobot Conversion

### Conversion Tool

LeRobot provides utilities to convert OpenX/RLDS datasets to LeRobotDataset format:

```bash
# Convert RLDS dataset to LeRobot format
python examples/port_datasets/openx_rlds.py \
    --raw-dir /path/to/bridge_orig/1.0.0 \
    --local-dir /path/to/local_dir \
    --repo-id your_username/bridge_dataset \
    --use-videos \
    --push-to-hub
```

### Conversion Process

The converter:
1. Reads RLDS episodes (TensorFlow datasets)
2. Extracts observations, actions, rewards, and metadata
3. Converts camera observations to MP4 videos
4. Organizes data into LeRobotDataset format (Parquet + videos + metadata)
5. Uploads to Hugging Face Hub

### Field Mapping

**RLDS to LeRobot mapping**:

| RLDS Field | LeRobot Field |
|------------|---------------|
| `observation` | `observation.state` and `observation.images.*` |
| `action` | `action` |
| `reward` | Not typically stored (learned via reward model) |
| `discount` | Not stored |
| `is_first` | Inferred from `frame_index == 0` |
| `is_last` | `next.done` |
| `is_terminal` | `next.done` |

**Note**: Some RLDS features may be omitted during conversion (e.g., depth streams, auxiliary action streams).

### Example Conversion Code

```python
# Minimal example for custom RLDS dataset conversion
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load RLDS dataset
rlds_dataset = tfds.load('libero_spatial_no_noops', split='train')

# Create LeRobot dataset
lerobot_dataset = LeRobotDataset.create(
    repo_id="your_username/libero_converted",
    fps=20,
    robot_type="panda",
    features={...}  # Define features based on RLDS schema
)

# Convert episodes
for episode_idx, episode in enumerate(rlds_dataset):
    for step_idx, step in enumerate(episode['steps']):
        # Extract data
        obs = step['observation']
        action = step['action']

        # Add to LeRobot dataset
        lerobot_dataset.add_frame({
            "observation.state": obs['state'],
            "observation.images.front": obs['image'],
            "action": action,
            "timestamp": step_idx / fps,
            "episode_index": episode_idx,
            "frame_index": step_idx,
            "next.done": step['is_last']
        })

    lerobot_dataset.save_episode(episode_idx)

# Finalize
lerobot_dataset.consolidate()
lerobot_dataset.push_to_hub()
```

### Performance Considerations

Large-scale conversions can be time-intensive:
- Converting 92,233 episodes: ~7 days on single node
- Solutions:
  - Multi-node parallelization
  - Episode-level sharding across compute nodes
  - Distributed computing frameworks

---

## Integration with Hugging Face Ecosystem

### Dataset Hosting

LeRobot datasets are hosted on Hugging Face Hub:
- Organization: https://huggingface.co/lerobot
- Community datasets: Public contributions from researchers
- Private datasets: Team/organization-specific data

### Accessing Datasets

```python
# Public dataset from Hub
dataset = LeRobotDataset("lerobot/aloha_static_coffee")

# Community-contributed dataset
dataset = LeRobotDataset("username/custom_robot_dataset")

# Private dataset (requires authentication)
from huggingface_hub import login
login()
dataset = LeRobotDataset("organization/private_dataset")
```

### Uploading Datasets

```python
# Push to Hub (requires authentication)
dataset.push_to_hub()

# Or via command-line after recording
python lerobot/scripts/record.py \
    --robot=koch \
    --repo-id=your_username/my_dataset \
    --push-to-hub
```

### Visualization Tools

LeRobot provides visualization utilities:

```bash
# Visualize dataset
lerobot-dataset-viz \
    --repo-id lerobot/aloha_static_coffee \
    --episode-index 0

# Visualize local dataset
lerobot-dataset-viz \
    --root /path/to/local/dataset \
    --local-files-only
```

**Web-based visualizer**: https://huggingface.co/spaces/lerobot/visualize_dataset

---

## Dependencies and Installation

### System Requirements

- **Python**: 3.10+
- **PyTorch**: 2.2+
- **Operating Systems**: Linux, macOS, Windows (with WSL)

### Installation Steps

#### 1. Create Virtual Environment

```bash
# Using miniconda
conda create -y -n lerobot python=3.10
conda activate lerobot
```

#### 2. Install FFmpeg

```bash
# Via conda
conda install ffmpeg -c conda-forge

# Or on Ubuntu/Debian
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

FFmpeg is required for video encoding/decoding. The installation typically provides ffmpeg 7.X with `libsvtav1` encoder.

#### 3. Install LeRobot

**From PyPI (basic)**:
```bash
pip install lerobot
```

**From PyPI (all features)**:
```bash
pip install 'lerobot[all]'
```

**From PyPI (specific features)**:
```bash
# For ALOHA simulation
pip install 'lerobot[aloha]'

# For PushT simulation
pip install 'lerobot[pusht]'

# For Feetech motors
pip install 'lerobot[feetech]'

# Multiple features
pip install 'lerobot[aloha,pusht,feetech]'
```

**From Source (for development)**:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

# Or with extras
pip install -e ".[all]"
```

#### 4. Optional: Experiment Tracking

```bash
# Weights & Biases
wandb login
```

### Core Dependencies

**Data and ML**:
- `torch >= 2.2`
- `datasets` (Hugging Face)
- `transformers` (Hugging Face)
- `numpy`
- `pillow`

**Video Processing**:
- `ffmpeg-python`
- `opencv-python`
- `pyav`

**Data Storage**:
- `pyarrow` (Parquet files)
- `h5py` (HDF5 support)

**Visualization**:
- `matplotlib`
- `rerun-sdk` (3D visualization)

**Robot Control** (optional):
- `gym-aloha` (ALOHA simulation)
- `gym-pusht` (PushT simulation)
- `dynamixel-sdk` (Dynamixel motors)
- `feetech-servo-sdk` (Feetech motors)

### Troubleshooting

**Linux build errors**:
```bash
sudo apt-get install cmake build-essential python3-dev pkg-config
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
```

**GPU support**:
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**RTX 50 series GPUs**: Ensure PyTorch version supports SM120 architecture (torch >= 2.8).

---

## Integration Approach for IL_RL_WorldModel Project

### Recommended Strategy

LeRobot provides an excellent foundation for the IL_RL_WorldModel project's data management needs:

1. **Data Collection Phase**:
   - Use LeRobotDataset format for recording demonstrations
   - Store multi-camera observations, proprioceptive state, actions, and language instructions
   - Leverage Hugging Face Hub for dataset versioning and sharing

2. **Imitation Learning Pre-training**:
   - Load demonstrations via `LeRobotDataset`
   - Use `delta_timestamps` for temporal context (observation stacks)
   - Integrate with PyTorch DataLoader for efficient batching

3. **RL Training Phase**:
   - Store RL episodes in LeRobotDataset format for consistency
   - Maintain separate replay buffers for real vs. imagined experiences
   - Use episode-level metadata to track data source (human demo, RL episode, world model rollout)

4. **World Model Training**:
   - Access historical frames via `delta_timestamps` for sequence modeling
   - Train on mixed data: demonstrations + RL episodes
   - Support video prediction by accessing image sequences

### Code Integration Example

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

class VLADataModule:
    """Data module for VLA + RL training."""

    def __init__(self, demo_dataset_id, rl_dataset_path, config):
        # Load demonstration dataset
        self.demo_dataset = LeRobotDataset(
            demo_dataset_id,
            delta_timestamps={
                "observation.images.top": [-0.1, 0.0],  # Current + 1 previous frame
                "observation.state": [0.0],
                "action": [0.0]
            }
        )

        # Load RL experience dataset
        self.rl_dataset = LeRobotDataset(
            rl_dataset_path,
            local_files_only=True,
            delta_timestamps={
                "observation.images.top": [-0.1, 0.0],
                "observation.state": [0.0],
                "action": [0.0, 0.1, 0.2, 0.3]  # Current + 3 future actions
            }
        )

        self.batch_size = config.batch_size

    def train_dataloader(self):
        # Combine demonstration and RL data
        combined_dataset = ConcatDataset([self.demo_dataset, self.rl_dataset])

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
```

### Task-Conditioned Policy Integration

```python
# Load dataset with task information
dataset = LeRobotDataset("lerobot/aloha_mobile_tasks")

# Access sample with task
sample = dataset[0]
task_index = sample["task_index"].item()

# Task descriptions available in metadata
# Use for language-conditioned VLA policy:
# policy_output = vla_model(
#     images=sample["observation.images.top"],
#     state=sample["observation.state"],
#     instruction=task_description
# )
```

### World Model Data Access

```python
# Configure for video prediction
world_model_dataset = LeRobotDataset(
    "your_username/rl_experience",
    delta_timestamps={
        # 10 frames of history for context
        "observation.images.top": [t / 30 for t in range(-10, 0)],
        # Current action
        "action": [0.0]
    }
)

# Each sample provides 10-frame history + action for predicting next frame
for sample in world_model_dataset:
    history_images = sample["observation.images.top"]  # [10, C, H, W]
    action = sample["action"]  # [1, action_dim]

    # Train world model: predict next frame given history + action
    # next_frame_pred = world_model(history_images, action)
```

---

## Example Datasets

### Available on Hugging Face Hub

- **lerobot/aloha_static_coffee**: ALOHA robot static manipulation
- **lerobot/aloha_mobile_tasks**: ALOHA mobile manipulation with multiple tasks
- **lerobot/pusht**: PushT simulation dataset
- **lerobot/xarm_lift_medium**: xArm manipulation tasks

### Dataset Statistics

Typical dataset sizes:
- Episodes: 50-1000+ episodes per dataset
- Episode length: 100-500 steps
- Camera resolution: 480x640 to 1080x1920
- FPS: 10-30 Hz
- File size: 1-50 GB per dataset (depending on video quality and episode count)

---

## Key Takeaways

1. **LeRobotDataset** provides a standardized, efficient format for robot learning data combining Parquet (tabular), MP4 (video), and JSON (metadata).

2. **RLDS** is the standard format for TensorFlow-based RL datasets, with converters available to LeRobot format.

3. **Delta timestamps** enable powerful temporal queries for observation stacks and action sequences without manual indexing.

4. **Hugging Face integration** provides seamless dataset hosting, versioning, and sharing infrastructure.

5. **Language conditioning** is natively supported via task descriptions in metadata, ideal for VLA models.

6. **PyTorch compatibility** ensures easy integration with modern deep learning workflows.

7. **Multi-modal data** (images, proprioception, actions, language) is cleanly organized with consistent naming conventions.

---

## References

- **LeRobot Documentation**: https://huggingface.co/docs/lerobot
- **LeRobot GitHub**: https://github.com/huggingface/lerobot
- **RLDS GitHub**: https://github.com/google-research/rlds
- **RLDS Blog Post**: https://research.google/blog/rlds-an-ecosystem-to-generate-share-and-use-datasets-in-reinforcement-learning/
- **LeRobotDataset v3.0 Blog**: https://huggingface.co/blog/lerobot-datasets-v3
- **Dataset Visualization Tool**: https://huggingface.co/spaces/lerobot/visualize_dataset
- **LeRobot Community**: https://huggingface.co/lerobot

---

**Last Updated**: 2025-10-18
