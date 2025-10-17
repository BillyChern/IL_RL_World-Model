# OpenPI Repository Exploration

## Repository Overview

**Repository**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
**License**: Apache 2.0
**Description**: Open-source collection of robotics models and packages from Physical Intelligence

### Key Models

1. **π₀ (pi0)** - Flow-based vision-language-action model (VLA)
2. **π₀-FAST** - Autoregressive VLA based on FAST action tokenizer
3. **π₀.₅ (pi0.5)** - Upgraded version with better open-world generalization trained with knowledge insulation

All models are pre-trained on 10,000+ hours of robot data.

---

## Repository Structure

```
openpi/
├── .github/
│   ├── CODEOWNERS
│   └── workflows/
│       ├── pre-commit.yml
│       └── test.yml
├── .vscode/
│   └── settings.json
├── docs/
│   ├── docker.md
│   ├── norm_stats.md
│   └── remote_inference.md
├── examples/
│   ├── inference.ipynb
│   ├── policy_records.ipynb
│   ├── convert_jax_model_to_pytorch.py
│   ├── aloha_real/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── compose.yml
│   │   ├── constants.py
│   │   ├── convert_aloha_data_to_lerobot.py
│   │   ├── env.py
│   │   ├── main.py
│   │   ├── real_env.py
│   │   ├── robot_utils.py
│   │   └── video_display.py
│   ├── aloha_sim/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── env.py
│   │   └── main.py
│   ├── droid/
│   │   ├── README.md
│   │   ├── compute_droid_nonidle_ranges.py
│   │   ├── convert_droid_data_to_lerobot.py
│   │   └── main.py
│   ├── libero/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── convert_libero_data_to_lerobot.py
│   │   └── main.py
│   ├── simple_client/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── main.py
│   └── ur5/
│       └── README.md
├── packages/
│   └── openpi-client/
│       ├── pyproject.toml
│       └── src/openpi_client/
├── scripts/
│   ├── compute_norm_stats.py
│   └── serve_policy.py
├── src/openpi/
│   ├── __init__.py
│   ├── conftest.py
│   ├── py.typed
│   ├── transforms.py
│   ├── transforms_test.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gemma.py
│   │   ├── gemma_fast.py
│   │   ├── lora.py
│   │   ├── lora_test.py
│   │   ├── model.py
│   │   ├── model_test.py
│   │   ├── pi0.py
│   │   ├── pi0_config.py
│   │   ├── pi0_fast.py
│   │   ├── pi0_test.py
│   │   ├── siglip.py
│   │   ├── tokenizer.py
│   │   ├── tokenizer_test.py
│   │   ├── vit.py
│   │   └── utils/
│   ├── models_pytorch/
│   │   ├── gemma_pytorch.py
│   │   ├── pi0_pytorch.py
│   │   ├── preprocessing_pytorch.py
│   │   └── transformers_replace/models/
│   ├── policies/
│   │   ├── aloha_policy.py
│   │   ├── droid_policy.py
│   │   ├── libero_policy.py
│   │   ├── policy.py
│   │   ├── policy_config.py
│   │   └── policy_test.py
│   ├── serving/
│   ├── shared/
│   │   └── download.py
│   └── training/
│       ├── checkpoints.py
│       ├── config.py
│       ├── data_loader.py
│       ├── data_loader_test.py
│       ├── droid_rlds_dataset.py
│       ├── optimizer.py
│       ├── sharding.py
│       ├── utils.py
│       ├── weight_loaders.py
│       └── misc/
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

---

## Model Architecture

### Pi0/Pi0.5 Architecture Overview

The pi0 model is a **diffusion-based vision-language-action model** that combines:

1. **Vision Encoder** (SigLIP/ViT)
2. **Language Model** (Gemma)
3. **Action Expert Network** (Gemma-based)
4. **Diffusion Action Decoder**

### Component Details

#### 1. Vision Encoder (SigLIP)

**File**: `src/openpi/models/siglip.py`

**Architecture**:
- Convolutional patch extraction with configurable patch size (e.g., 32x32)
- Positional embeddings (learned or sinusoidal 2D)
- Transformer encoder blocks with multi-head self-attention
- Layer normalization and GELU activation
- Optional pooling (single token, mean, or multi-head attention pooling)

**Variants**:
- Multiple scales: Ti, S, M, B, L, H, g, G, e
- Default for pi0: ViT-B with 768 width

**Input**: RGB images at standard resolution (224x224)
**Output**: Patch embeddings sequence

#### 2. Language Model (Gemma)

**File**: `src/openpi/models/gemma.py`

**Variants**:
- **Gemma 300M**: 1024 width, 18 depth, 8 heads (~311M parameters)
- **Gemma 2B**: 2048 width, 18 depth, 8 heads

**Architecture Components**:
- Vocabulary size: 257,152 tokens
- Rotary position embeddings (RoPE)
- Multi-head self-attention with grouped query attention
- Gated feed-forward networks with GELU activation
- RMSNorm layers with optional adaptive modulation
- Key-value caching for efficient inference
- Multi-expert support for mixture-of-experts

**Context**: Processes tokenized language instructions alongside visual inputs

#### 3. Pi0 Model Integration

**Files**:
- JAX: `src/openpi/models/pi0.py`
- PyTorch: `src/openpi/models_pytorch/pi0_pytorch.py`
- Config: `src/openpi/models/pi0_config.py`

**Core Components**:
```python
class Pi0:
    - PaliGemma (vision + language multimodal LLM)
    - Action expert network (Gemma-based)
    - action_in_proj: Projects actions to embedding dimension
    - action_out_proj: Projects embeddings back to action space
    - time_mlp_in/out: Timestep embedding (pi0.5 variant)
    - state_proj: State token projection (standard pi0)
```

**Configuration Parameters** (`Pi0Config`):
```python
dtype: "bfloat16"
paligemma_variant: "gemma_2b"
action_expert_variant: "gemma_300m"
action_dim: 32
action_horizon: 50
max_token_len: 200 (pi0.5) or 48 (pi0)
pi05: False  # Enable pi0.5-specific features
discrete_state_input: Auto-set based on pi05
```

**Input Specifications**:
- 3 RGB images (base, left wrist, right wrist) at IMAGE_RESOLUTION
- Image masks (boolean)
- State vector (dimension 32)
- Tokenized prompt (max 48 or 200 tokens)
- Prompt mask

**Output**:
- Actions: shape `[batch_size, action_horizon=50, action_dim=32]`

#### 4. Forward Pass Methods

**`embed_prefix()`**:
- Processes observation images and tokenized prompts
- Generates attention masks with full attention between image and language inputs
- Returns prefix embeddings

**`embed_suffix()`**:
- Embeds noisy actions and timesteps
- Uses sine-cosine positional encoding (range [0, 1])
- Two variants:
  - Pi0.5: Adaptive RMS normalization
  - Standard: MLP mixing

**`compute_loss()`**:
- Implements diffusion training via noise prediction
- Concatenates prefix/suffix tokens
- Applies attention masking
- Computes MSE between predicted and target noise

**`sample_actions()` (Inference)**:
- Iterative denoising (default 10 steps)
- Reverse diffusion: timestep progression from 1 to 0
- KV cache for efficiency
- Euler integration scheme

#### 5. Attention Mechanism

The `make_attn_mask()` function creates flexible attention patterns:
- Causal attention
- Prefix-LM attention
- Block-structured attention
- Cumulative masking support

---

## Model Variants and Checkpoints

### Available Pretrained Models

**Base Models** (pre-trained on 10k+ hours):
- `pi0_base`: Flow-based model
- `pi0_fast_base`: Autoregressive variant
- `pi05_base`: Knowledge-insulated version

**Checkpoint Locations**:
- Pi0 Base: `gs://openpi-assets/checkpoints/pi0_base/params`
- Pi0-FAST Base: `gs://openpi-assets/checkpoints/pi0_fast_base/params`
- Pi0.5 Base: `gs://openpi-assets/checkpoints/pi05_base/params`

### Fine-Tuned Specialists

**DROID (table-top manipulation)**:
- `pi0_droid`
- `pi0_fast_droid`
- `pi05_droid`

**ALOHA (dual-arm manipulation)**:
- `pi0_aloha` / `pi05_aloha`
- `pi0_aloha_pen_uncap` / `pi05_aloha_pen_uncap`
- Task-specific: towel folding, container unpacking

**LIBERO (simulation benchmark)**:
- `pi0_libero` / `pi0_fast_libero` / `pi05_libero`
- LoRA variants for efficient fine-tuning
- State-of-the-art benchmark performance

---

## Training Configuration

### File: `src/openpi/training/config.py`

**30+ Predefined Configurations** organized by use case:

**Inference Configs**:
- `pi0_aloha`, `pi05_aloha`
- `pi0_droid`, `pi0_fast_droid`, `pi05_droid`
- `pi0_libero`, `pi05_libero`

**Fine-tuning Configs**:
- Libero: `pi0_libero`, `pi0_fast_libero`, `pi05_libero` (with LoRA variants)
- Aloha: `pi0_aloha_pen_uncap`, `pi05_aloha_pen_uncap`
- DROID: `pi0_fast_full_droid_finetune`, `pi05_full_droid_finetune`
- Simulation: `pi0_aloha_sim`

**Debugging Configs**:
- `debug`, `debug_restore`, `debug_pi05`

### Config Loading Methods

**Command-line**:
```python
def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(...)
```

**Programmatic**:
```python
def get_config(config_name: str) -> TrainConfig:
    """Get a config by name with fuzzy-matching suggestions"""
```

### Weight Loading

**File**: `src/openpi/training/weight_loaders.py`

**Weight Loader Protocol**:
- `load()` method accepts and returns model parameters
- Merges loaded weights into existing parameters

**Checkpoint Sources**:
1. Trained checkpoints: `./checkpoints/<config>/<exp>/<step>/params`
2. Released checkpoints: `gs://openpi-assets/checkpoints/<model>/params`

**Loader Implementations**:
- `NoOpWeightLoader`: Returns parameters unchanged
- `CheckpointWeightLoader`: Loads complete weight sets, merges LoRA weights
- `PaliGemmaWeightLoader`: Loads official PaliGemma checkpoints from Vertex Model Garden

**Cache Management** (`src/openpi/shared/download.py`):
- Storage: `~/.cache/openpi` (configurable via `OPENPI_DATA_HOME`)
- Atomic downloads with `.partial` files and file locking
- Cache expiration policy with timestamp-based invalidation
- Consistent permissions for shared environments

---

## Robot Support and Action Spaces

### Supported Robot Platforms

#### 1. DROID Platform

**Files**:
- Policy: `src/openpi/policies/droid_policy.py`
- Example: `examples/droid/`

**Action Space**: 8 dimensions
- 7 joint positions
- 1 gripper position

**State Representation**: 8 elements
- 7 joint positions from `observation/joint_position`
- 1 gripper position from `observation/gripper_position`

**Input Images**:
- Base/exterior camera: 224x224x3 (uint8)
- Wrist camera: 224x224x3 (uint8)
- Automatic conversion from float32 (C,H,W) to uint8 (H,W,C)

**Image Mappings**:
- PI0/PI05: Three named images with selective masking (right wrist masked)
- PI0_FAST: Three named images without masking

**Capabilities**: Simple table-top manipulation (pick-and-place)

**Control Architecture**: Remote policy server with 0.5-1 sec latency per chunk

#### 2. ALOHA Platform

**Files**:
- Policy: `src/openpi/policies/aloha_policy.py`
- Example: `examples/aloha_real/`, `examples/aloha_sim/`

**Action Space**: 14 dimensions
- Left arm: 6 joint angles + 1 gripper
- Right arm: 6 joint angles + 1 gripper

**State Representation**: 14 elements
```
[left_arm_joints(6), left_gripper(1), right_arm_joints(6), right_gripper(1)]
```

**Actions**: `[action_horizon, 14]`

**Input Cameras**:
- `cam_high`: Top-down view
- `cam_low`: Front view
- `cam_left_wrist`: Left end-effector
- `cam_right_wrist`: Right end-effector

**Image Format**:
- Input: `[channel, height, width]` arrays (uint8, 0-255)
- Converted to `[height, width, channel]` internally
- Missing cameras replaced with black images

**Image Mappings**:
- Base: `base_0_rgb`
- Left wrist: `left_wrist_0_rgb`
- Right wrist: `right_wrist_0_rgb`

**Gripper Transform**: Converts between Aloha's linear gripper space and angular space via inverse kinematics

**Tasks**: Toast retrieval, towel folding, tupperware opening

#### 3. UR5 Platform

**Files**: `examples/ur5/`

**Action Space**: 7 dimensions
- 6 DoF (joints)
- 1 gripper control

**State**: Concatenated joint positions + gripper value

**Sensors**:
- Base RGB camera
- Left wrist RGB camera
- Right wrist slot (unused, filled with zeros)

**Action Processing**: Converts absolute joint actions to delta actions for first 6 dimensions, preserves gripper action

**Integration Pattern**: Requires defining input/output transforms, data config, and training config

#### 4. ARX X5 / Custom Robot Support

**Status**: No direct ARX X5 support found in repository

**Integration Guide** (from UR5 example):

Custom robot integration requires:

1. **Input/Output Transform Classes**:
   - Define camera mappings
   - State/action space transformations
   - Convert to expected format (3 images: base, left wrist, right wrist)

2. **Data Configuration Factory**:
   - Specify action dimensions
   - Define normalization parameters
   - Configure state representations

3. **Training Configuration**:
   - Select model variant
   - Specify dataset
   - Load/compute normalization statistics

**Key Considerations**:
- Action space: Recommend 7-8 dimensions (joints + gripper)
- State space: Match action dimensions
- Vision: 1-3 RGB cameras (224x224)
- Delta vs absolute actions: Use delta for transfer learning

---

## Data Input/Output Formats

### Input Data Format

**General Structure**:
```python
{
    "observation": {
        "images": {
            "base_0_rgb": np.ndarray,      # [H, W, C], uint8
            "left_wrist_0_rgb": np.ndarray, # [H, W, C], uint8
            "right_wrist_0_rgb": np.ndarray # [H, W, C], uint8 (optional)
        },
        "state": np.ndarray,  # [state_dim], typically 7-14
    },
    "action": np.ndarray,     # [action_horizon, action_dim]
    "prompt": str or bytes    # Language instruction (UTF-8)
}
```

**Image Requirements**:
- Format: uint8, range [0, 255]
- Shape: [height, width, channels] (HWC format)
- Standard size: 224x224x3
- Color space: RGB

**State Requirements**:
- Proprioceptive information (joint angles, gripper position)
- Dimension: Typically 7-32 elements
- Normalized during training/inference

**Action Requirements**:
- Shape: [action_horizon, action_dim]
- Default: [50, 32] for base models
- Robot-specific: [50, 8] for DROID, [50, 14] for ALOHA, [50, 7] for UR5
- Can be absolute or delta (relative) positions

**Language Prompts**:
- Free-form text instructions
- Tokenized using Paligemma tokenizer
- Max length: 48 tokens (pi0) or 200 tokens (pi0.5)
- Examples: "pick up the cup", "fold the towel"

### Output Data Format

**Inference Output**:
```python
{
    "actions": np.ndarray,  # [action_horizon, action_dim]
    "policy_time": float,   # Inference latency (seconds)
}
```

**Action Output**:
- Same shape as input actions
- Denormalized to robot's action space
- Ready for direct robot control

### Data Transformation Pipeline

**File**: `src/openpi/transforms.py`

**Transform Operations**:

1. **Structural Transforms**:
   - `RepackTransform`: Reorganizes dictionary structure with "/" separators
   - `CompositeTransform`: Chains multiple transforms sequentially
   - `Group`: Organizes into input/output categories

2. **Normalization**:
   - `Normalize`: Z-score or quantile-based normalization
   - `Unnormalize`: Reverses normalization with dimension padding

3. **Action Space**:
   - `DeltaActions`: Converts absolute to delta space using state
   - `AbsoluteActions`: Reverses delta conversion
   - `SubsampleActions`: Reduces sequences by stride

4. **Image Processing**:
   - `ResizeImages`: Padding-aware resizing

5. **Tokenization**:
   - `TokenizePrompt`: Converts text with optional state integration
   - `TokenizeFASTInputs`: Generates tokens for FAST models
   - `ExtractFASTActions`: Decodes tokens to actions

6. **Data Enhancement**:
   - `InjectDefaultPrompt`: Inserts default prompts
   - `PromptFromLeRobotTask`: Extracts prompts from task indices
   - `PadStatesAndActions`: Zero-pads to model dimensions

**Pipeline Sequence**:
```
Input Data
  -> Repack transforms
  -> Default prompt injection
  -> Data-specific transforms
  -> Normalization
  -> Model-specific transforms
  -> Policy inference
  -> Denormalization
  -> Output transforms
  -> Repack transforms
Output Data
```

### Normalization Statistics

**File**: `docs/norm_stats.md`

**Purpose**: Normalize proprioceptive state inputs and action targets

**Storage**: Stored alongside model checkpoint

**Computation**: Derived from training data

**Fine-tuning Strategy**:
- **Reload existing stats**: When hardware matches pre-training (more "familiar" to model)
- **Compute new stats**: When significant domain differences exist
- **Recommendation**: Try both empirically

**Critical Requirement**: Maintain consistency with pre-training action space definitions

### Dataset Loading

**File**: `src/openpi/training/data_loader.py`

**Dataset Sources**:

1. **LeRobot Datasets**:
   - Real robot data via `create_torch_dataset()`
   - Task-based prompting with `PromptFromLeRobotTask`

2. **RLDS/DROID Datasets**:
   - Alternative source via `create_rlds_dataset()`
   - Robotics learning from demonstrations

3. **Fake Datasets**:
   - Synthetic data for testing via `FakeDataset`
   - Random observations/actions matching specs

**Batch Preparation**:
- Individual sample processing through composed transforms
- Batched processing for pre-batched RLDS data
- `_collate_fn` standardizes batch creation with numpy arrays

**Framework Support**:
- JAX and PyTorch backends
- Framework-specific sharding strategies
- PyTorch DDP integration via `DistributedSampler`

---

## Dependencies and Requirements

### System Requirements

**Operating System**: Ubuntu 22.04 (tested)

**GPU Requirements**:

| Task | Memory | Example GPU |
|------|--------|-------------|
| Inference | >8 GB | RTX 4090 |
| LoRA Fine-Tuning | >22.5 GB | RTX 4090 |
| Full Fine-Tuning | >70 GB | A100/H100 |

### Core Dependencies

**From `pyproject.toml`**:

**Python**: >=3.11

**JAX Ecosystem**:
- `jax[cuda12]==0.5.3`
- `flax==0.10.2`
- `equinox>=0.11.8`

**PyTorch**:
- `torch==2.7.1`
- `transformers==4.53.2`

**Data Handling**:
- `numpy`
- `polars`
- `dm-tree`
- `flatbuffers`

**Computer Vision**:
- `opencv-python`
- `pillow`
- `imageio`

**Robotics**:
- `gym-aloha`
- `lerobot` (from GitHub)
- `openpi-client`

**Utilities**:
- `einops`
- `jaxtyping`
- `tyro`
- `wandb`
- `rich`

**Optional (RLDS)**:
- `dlimp` (from GitHub)
- `tensorflow-cpu==2.15.0`
- `tensorflow-datasets==4.9.9`

**Development**:
- `pytest`
- `ruff`
- `pre-commit`
- `ipykernel`
- `matplotlib`
- `pynvml`

### Installation

**Standard Installation**:

```bash
# Install uv package manager
pip install uv

# Clone repository with submodules
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Docker Installation**:

See `docs/docker.md` for containerized setup

---

## Using Pi0.5 in Our Project

### Integration Strategy

Based on the CLAUDE.md project architecture, here's how to integrate pi0.5:

#### 1. VLA Policy Module Implementation

**Initialize from OpenPI**:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# Load pi0.5 configuration
config = _config.get_config("pi05_droid")  # or pi05_aloha

# Download checkpoint (cached to ~/.cache/openpi)
checkpoint_path = download.maybe_download(config.checkpoint_path)

# Load policy with pretrained weights
policy = policy_config.load_policy_from_config(
    config=config,
    checkpoint_path=checkpoint_path,
    default_prompt="pick up the object",  # optional
    pytorch_device="cuda:0"  # or "cpu"
)
```

**Inference**:

```python
# Prepare observation
obs = {
    "observation": {
        "images": {
            "base_0_rgb": base_image,      # [224, 224, 3], uint8
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8)
        },
        "state": robot_state  # [8] for DROID-like robot
    },
    "prompt": "pick up the red cube"
}

# Run inference
output = policy.infer(obs)
actions = output["actions"]  # [action_horizon, action_dim]
```

#### 2. Adapting to Custom Robot (ARX X5)

**Create Custom Policy Module** (following UR5 pattern):

```python
# custom_robot_policy.py

import numpy as np
from openpi import transforms as _transforms
from openpi.training import data_loader

class CustomRobotInputs:
    """Transform raw robot data to pi0.5 format"""

    def __call__(self, data: dict) -> dict:
        # Extract state (adapt to your robot's dimensions)
        state = np.concatenate([
            data["observation/joint_position"],  # [6 or 7]
            data["observation/gripper_position"]  # [1]
        ])

        # Process images to uint8 HWC format
        base_img = self._to_uint8_hwc(data["observation/exterior_image_1"])
        wrist_img = self._to_uint8_hwc(data["observation/wrist_image"])

        # Create pi0.5-compatible observation
        return {
            "observation": {
                "images": {
                    "base_0_rgb": base_img,
                    "left_wrist_0_rgb": wrist_img,
                    "right_wrist_0_rgb": np.zeros_like(wrist_img)
                },
                "state": state
            },
            "action": data.get("action"),  # [action_horizon, action_dim]
            "prompt": data.get("prompt", "")
        }

    def _to_uint8_hwc(self, img):
        """Convert to uint8 HWC format"""
        if img.ndim == 3 and img.shape[0] == 3:  # CHW
            img = np.transpose(img, (1, 2, 0))  # -> HWC
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        return img

class CustomRobotOutputs:
    """Transform pi0.5 outputs to robot commands"""

    def __call__(self, data: dict) -> dict:
        # Extract first N actions for your robot
        actions = data["actions"][:, :7]  # Example: 7-DoF robot

        # Optional: convert delta to absolute actions
        # actions = self._delta_to_absolute(actions, data["observation/state"])

        return {"actions": actions}
```

**Configure Data Loading**:

```python
from openpi.training.data_loader import DataConfig

def custom_robot_data_config() -> DataConfig:
    return DataConfig(
        action_dim=7,  # Adjust for your robot
        state_dim=7,   # Adjust for your robot
        action_horizon=50,
        input_transform=CustomRobotInputs(),
        output_transform=CustomRobotOutputs(),
        # Compute normalization stats from your data
        norm_stats=compute_norm_stats(your_dataset),
        # Or reload from similar robot
        # assets_config=AssetsConfig(checkpoint_dir="gs://...", norm_id="...")
    )
```

#### 3. RL Fine-Tuning Integration

**Combine with HIL-SERL Approach**:

```python
# rl_training.py

from openpi.training import config as train_config
from openpi.training import optimizer, data_loader

# Start with pre-trained pi0.5
base_config = train_config.get_config("pi05_base")

# Configure for fine-tuning
custom_config = base_config.replace(
    # Your dataset
    data=custom_robot_data_config(),

    # LoRA for efficient fine-tuning
    model=base_config.model.replace(
        paligemma_lora_rank=32,
        action_expert_lora_rank=32
    ),

    # RL-specific training params
    optimizer=optimizer.OptimizerConfig(
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=100
    ),

    # Checkpoint saving
    checkpoint_dir="./checkpoints/custom_robot_rl",
    log_wandb=True,
    wandb_project="il_rl_worldmodel"
)

# Training loop will combine:
# - Real robot experience from replay buffer
# - Imagined experience from world model
# - Imitation loss on demonstrations
# - RL loss on rewards
```

**Hybrid Loss Function**:

```python
def compute_hybrid_loss(policy, real_batch, imagined_batch, demo_batch, λ=0.1):
    """L = L_RL + λ * L_IL"""

    # RL loss on real + imagined data
    rl_loss_real = compute_diffusion_loss(policy, real_batch)
    rl_loss_imagined = compute_diffusion_loss(policy, imagined_batch)
    rl_loss = rl_loss_real + rl_loss_imagined

    # Imitation loss on demonstrations
    il_loss = compute_diffusion_loss(policy, demo_batch)

    # Combined loss
    total_loss = rl_loss + λ * il_loss

    return total_loss
```

#### 4. World Model Integration

**Use Pi0.5 for Policy, Separate World Model**:

```python
# world_model.py

class WorldModel:
    """DreamerV3-style or video prediction model"""

    def __init__(self):
        self.encoder = CNNEncoder()
        self.dynamics = GRU()  # or Transformer
        self.decoder = MLPDecoder()
        self.reward_model = MLPReward()

    def predict_next(self, state, action, instruction):
        """Predict next state and reward"""
        latent = self.encoder(state)
        next_latent = self.dynamics(latent, action, instruction)
        next_state = self.decoder(next_latent)
        reward = self.reward_model(next_latent)
        return next_state, reward

    def imagine_trajectory(self, initial_state, policy, horizon=10):
        """Generate imagined rollout"""
        states, actions, rewards = [], [], []
        state = initial_state

        for _ in range(horizon):
            # Policy generates action
            action = policy.infer(state)["actions"][0]

            # World model predicts outcome
            next_state, reward = self.predict_next(state, action, instruction)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return states, actions, rewards

# Training pipeline
world_model = WorldModel()
policy = load_pi05_policy()

# Asynchronous loop
while training:
    # Real robot collection
    real_experience = collect_from_robot(policy)
    replay_buffer.add(real_experience)

    # Update world model
    world_model.train(replay_buffer)

    # Generate imagined rollouts
    imagined_experience = []
    for state in replay_buffer.sample_states():
        traj = world_model.imagine_trajectory(state, policy)
        imagined_experience.append(traj)

    # Train policy on mixed data
    real_batch = replay_buffer.sample()
    imagined_batch = imagination_buffer.sample()
    loss = compute_hybrid_loss(policy, real_batch, imagined_batch)
    update_policy(policy, loss)
```

#### 5. Remote Inference for Real-Time Control

**Server Setup** (on GPU machine):

```bash
# Start policy server
python scripts/serve_policy.py \
    --policy pi05_droid \
    --port 8000 \
    --default_prompt "perform manipulation task"
```

**Client Integration** (on robot control computer):

```python
from openpi_client import WebsocketPolicyClient

# Connect to policy server
client = WebsocketPolicyClient(
    host="192.168.1.100",  # GPU server IP
    port=8000
)

# Control loop
while True:
    # Capture observation
    obs = robot.get_observation()

    # Query policy (0.5-1s latency)
    output = client.infer(obs)
    actions = output["actions"]

    # Execute first action
    robot.execute_action(actions[0])

    # Human intervention check
    if human_intervenes():
        corrected_action = human_input()
        replay_buffer.add(obs, corrected_action, reward=1.0)
```

### Key Configuration Parameters

**Model Selection**:
- `pi05_base`: General-purpose pre-trained model
- `pi05_droid`: Fine-tuned for table-top manipulation
- `pi05_aloha`: Fine-tuned for dual-arm tasks

**Fine-Tuning Options**:
- LoRA rank: 16 or 32 (efficient adaptation)
- Full fine-tuning: All parameters (requires >70GB GPU)

**Action Space Adaptation**:
- Default: 32 dimensions
- Robot-specific: 7-14 dimensions (specify in data config)
- Use delta actions for better transfer learning

**Normalization Strategy**:
- Reload stats: When robot similar to pre-training
- Compute new: When robot significantly different
- Critical: Match pre-training action space definitions

### Advantages for Our Project

1. **Pre-trained VLA**: 10,000+ hours of robot data
2. **Language Conditioning**: Natural task specification
3. **Both JAX and PyTorch**: Flexibility for integration
4. **Proven Performance**: SOTA on multiple benchmarks
5. **Efficient Fine-Tuning**: LoRA support for limited compute
6. **Remote Inference**: Separates control and computation
7. **Active Development**: Regular updates from Physical Intelligence

### Limitations and Considerations

1. **No Native ARX X5 Support**: Requires custom integration
2. **Large Model Size**: 2B+ parameters (inference latency)
3. **GPU Memory**: Significant requirements for training
4. **Action Horizon**: Fixed 50-step horizon (may need adaptation)
5. **Camera Requirements**: Expects 2-3 RGB cameras
6. **Proprietary Checkpoints**: Hosted on Google Cloud Storage

---

## Additional Resources

**Documentation**:
- Main README: [openpi/README.md](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)
- Docker setup: [docs/docker.md](https://github.com/Physical-Intelligence/openpi/blob/main/docs/docker.md)
- Normalization: [docs/norm_stats.md](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)
- Remote inference: [docs/remote_inference.md](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)

**Examples**:
- Inference notebook: [examples/inference.ipynb](https://github.com/Physical-Intelligence/openpi/blob/main/examples/inference.ipynb)
- ALOHA real robot: [examples/aloha_real/](https://github.com/Physical-Intelligence/openpi/tree/main/examples/aloha_real)
- DROID setup: [examples/droid/](https://github.com/Physical-Intelligence/openpi/tree/main/examples/droid)
- UR5 integration: [examples/ur5/](https://github.com/Physical-Intelligence/openpi/tree/main/examples/ur5)

**Key Papers**:
- π₀ (Physical Intelligence blog/papers)
- π₀.₅ with knowledge insulation
- PaliGemma (Google vision-language model)
- SigLIP (vision encoder)
- Gemma (language model)

**Contact**:
- Repository: https://github.com/Physical-Intelligence/openpi
- Issues: Use GitHub Issues for bug reports and questions
- License: Apache 2.0

---

## Summary

The OpenPI repository provides a comprehensive, production-ready implementation of vision-language-action models for robotic manipulation. The pi0.5 model represents the state-of-the-art in VLA approaches with:

- Strong pre-training on diverse robot data
- Flexible language-based task specification
- Efficient fine-tuning capabilities
- Multi-platform support (DROID, ALOHA, UR5)
- Active maintenance and development

For integration into our IL+RL+WorldModel project:
1. Use pi0.5 as the pre-trained VLA policy module
2. Adapt to custom robot through policy transform classes
3. Fine-tune with LoRA on robot-specific demonstrations
4. Integrate with RL training loop (HIL-SERL approach)
5. Combine with separate world model for imagined rollouts
6. Deploy with remote inference for real-time control

The repository's modular design and comprehensive examples make it an excellent foundation for our research project.
