# System Architecture

**IL + RL + World Model for Robotic Manipulation**

This document describes the complete system architecture integrating Vision-Language-Action models, Human-in-the-Loop Reinforcement Learning, and World Models for sample-efficient robot manipulation.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Training Pipeline](#training-pipeline)
- [Hardware Architecture](#hardware-architecture)
- [Software Stack](#software-stack)
- [Component Interactions](#component-interactions)
- [Implementation Plan](#implementation-plan)

## Overview

### Core Innovation

The system combines three complementary approaches:

1. **π0.5 VLA (Vision-Language-Action)**
   - Pre-trained on 10,000+ hours of robot data
   - Provides broad generalization via language conditioning
   - Initializes policy with "common sense" knowledge

2. **HIL-SERL (Human-in-the-Loop RL)**
   - Real-time Soft Actor-Critic with demonstrations
   - Human interventions for safety and corrective guidance
   - Achieves 2x improvement over pure imitation learning

3. **DreamerV3 (World Model)**
   - Latent dynamics model (RSSM architecture)
   - Generates 10+ imagined rollouts per real robot trial
   - Reduces real-world data requirements by 5-10x

### Key Metrics

- **Training efficiency**: ~10 imagined experiences per real experience
- **Intervention rate**: 30% initially → 0% at convergence
- **Data mixing**: 80% real / 20% imagined → 50/50 or beyond
- **Imagination horizon**: 5-10 steps initially → 15+ steps
- **Training time**: 1-2 hours for complex bi-manual tasks (projected)

## System Components

### 1. VLA Policy Module (`models/vla/`)

**Architecture**:
- **Vision Encoder**: SigLIP ViT (frozen or lightly fine-tuned)
- **Language Model**: Gemma 2B (frozen)
- **Action Expert**: Gemma 300M (trainable)
- **Diffusion Decoder**: 10-step iterative denoising for actions

**Inputs**:
- RGB images: 3 cameras (left, right, base) @ 224x224
- Proprioception: 14D state (dual ARX X5: 2×(6 joints + gripper))
- Language instruction: Natural language string (e.g., "fold the yellow towel")

**Outputs**:
- Action sequence: [action_horizon, 14] (50 timesteps × 14 dimensions)
- Per-arm: 7D (6 joint positions + gripper state)

**Key Features**:
- Pre-trained weights from OpenPI
- Language-conditioned multi-task policy
- Action chunking (50-step horizon)
- Delta action prediction

**Implementation**: `models/vla/pi05_policy.py`

---

### 2. World Model (`models/world_model/`)

**Architecture** (DreamerV3 RSSM):

```
Input: (observation_t, action_t, instruction)
       ↓
[Vision Encoder] → image features (2048-dim)
       ↓
[RSSM Dynamics]
   ├─ Deterministic state: GRU (8192-dim)
   ├─ Stochastic state: 32 categorical × 64 classes = 2048-dim
   └─ Prior/Posterior networks
       ↓
[Decoders]
   ├─ Image Decoder: Reconstructs next observation
   ├─ Reward Predictor: Predicts reward (2-hot encoding)
   └─ Continue Predictor: Predicts episode termination
       ↓
Output: (predicted_observation_{t+1}, predicted_reward_t, done_t)
```

**Training Objectives**:
1. **Prediction loss**: Reconstruct observations and rewards
2. **Dynamics loss**: Learn accurate state transitions
3. **Representation loss**: KL divergence for stochastic states

**Imagination Rollouts**:
```python
# Sample initial real state from replay buffer
state_0 = sample_from_buffer()

# Unroll world model for H steps
for t in range(imagination_horizon):
    action_t = policy(state_t, instruction)
    state_{t+1}, reward_t = world_model(state_t, action_t)

# Store imagined transitions in imagination buffer
```

**Key Features**:
- Faster-than-real-time simulation (GPU-accelerated)
- Language-conditioned dynamics
- Uncertainty estimation via stochastic latents
- Continuous online updates

**Implementation**: `models/world_model/dreamer_v3.py`

---

### 3. Reward Classifier (`models/reward/`)

**Architecture**:
- **Encoder**: ResNet-10 (pre-trained on ImageNet)
- **Classifier Head**: MLP → sigmoid output
- **Loss**: Binary cross-entropy

**Training Data**:
- Positive samples: Successful demo end states
- Negative samples: Failure/incomplete trajectories

**Input**: Current observation (image + proprioception)
**Output**: Reward ∈ [0, 1] (success probability)

**Key Features**:
- Sparse reward signal for RL
- Task-agnostic (learns from demos)
- Data augmentation for robustness

**Implementation**: `models/reward/reward_classifier.py`

---

### 4. RL Training Pipeline (`training/rl/`)

**Algorithm**: Soft Actor-Critic (SAC)

**Components**:
```
Policy π(a|s, instruction)       [Actor]
├─ Vision encoder (from VLA)
├─ Language encoder (from VLA)
└─ Action head (trainable)

Q(s, a, instruction)             [Critic - Ensemble of 2]
├─ State-action encoder
└─ Q-value head

α (temperature)                   [Automatic entropy tuning]
```

**Hybrid Loss Function**:
```
L_total = L_RL + λ(t) * L_IL

where:
  L_RL = Actor loss + Critic loss + α loss (standard SAC)
  L_IL = Behavioral cloning loss on demonstrations
  λ(t) = exponential decay (1.0 → 0.1 over training)
```

**Replay Buffer Structure**:
```python
class ReplayBuffer:
    # Real robot experiences
    observations: [N, T, obs_dim]
    actions: [N, T, action_dim]
    rewards: [N, T]
    dones: [N, T]
    instructions: [N, instruction_dim]
    intervention_flags: [N, T]  # Track human interventions

    # Demonstration subset
    demo_indices: List[int]

    # Sampling strategy
    sample_ratio = {
        'demos': 0.5,
        'online': 0.3,
        'interventions': 0.2
    }
```

**Key Features**:
- Initialize from VLA pre-trained weights
- Freeze/partially freeze vision & language encoders
- 50/50 sampling: demos + online data
- Prioritize intervention transitions

**Implementation**: `training/rl/sac_trainer.py`

---

### 5. Human Intervention System (`robot/teleop/`)

**Intervention Detection**:
```python
if external_command_received():
    # Human takes control
    intervention_mode = True
    commanded_actions = keyboard_input()

    # Execute human command
    robot.execute(commanded_actions)

    # Log intervention transition
    buffer.add(
        obs=current_obs,
        action=commanded_actions,
        reward=reward,
        done=done,
        intervention=True  # Mark for prioritized replay
    )
```

**Keyboard Control Mapping**:
```
Left Arm:
  W/S: Joint 1 ±
  A/D: Joint 2 ±
  Q/E: Joint 3 ±
  Z/C: Joint 4 ±
  R/F: Joint 5 ±
  T/G: Joint 6 ±
  Space: Toggle gripper

Right Arm:
  I/K: Joint 1 ±
  J/L: Joint 2 ±
  U/O: Joint 3 ±
  N/M: Joint 4 ±
  Y/H: Joint 5 ±
  P/;: Joint 6 ±
  Enter: Toggle gripper
```

**Implementation**: `robot/teleop/keyboard_intervention.py`

---

### 6. ARX X5 Robot Interface (`robot/arx_x5/`)

**Control Loop** (50Hz):
```python
class ARXController:
    def __init__(self):
        self.left_arm = SingleArm(port='/dev/ttyUSB0')
        self.right_arm = SingleArm(port='/dev/ttyUSB1')
        self.control_freq = 50  # Hz

    def step(self, action: np.ndarray) -> Observation:
        # action shape: [14] (7 per arm)
        left_action = action[:7]   # 6 joints + gripper
        right_action = action[7:]

        # Send joint commands
        self.left_arm.set_joint_positions(left_action[:6])
        self.left_arm.set_gripper(left_action[6])

        self.right_arm.set_joint_positions(right_action[:6])
        self.right_arm.set_gripper(right_action[6])

        # Read proprioception
        obs = self.get_observation()
        return obs

    def get_observation(self) -> Dict:
        return {
            'joint_positions': np.array([...]),  # 12D
            'joint_velocities': np.array([...]), # 12D
            'joint_currents': np.array([...]),   # 12D (torque proxy)
            'gripper_states': np.array([...]),   # 2D
            'images': {
                'left': camera_left.read(),      # 224x224x3
                'right': camera_right.read(),
                'base': camera_base.read(),
            }
        }
```

**Implementation**: `robot/arx_x5/controller.py`

---

## Data Flow

### Training Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DEMONSTRATION DATA                        │
│                   (RLDS Format via LeRobot)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├──────────────────────────────────────────┐
                 │                                          │
                 ▼                                          ▼
    ┌────────────────────────┐              ┌──────────────────────────┐
    │  VLA Pre-training      │              │  Reward Classifier       │
    │  (Imitation Learning)  │              │  Training                │
    └────────┬───────────────┘              └──────────┬───────────────┘
             │                                         │
             │ Pretrained VLA                          │ Reward Model
             ▼                                         ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                 ASYNCHRONOUS TRAINING LOOP                      │
    │                                                                  │
    │  ┌──────────────────┐         ┌───────────────────────────┐   │
    │  │  Robot           │         │  GPU Training             │   │
    │  │  Collection      │ ──────► │  (Separate GPU/CPU)      │   │
    │  │  Thread          │ Real    │                           │   │
    │  │                  │ Data    │  ┌────────────────────┐  │   │
    │  │  - Policy exec   │         │  │  Replay Buffer     │  │   │
    │  │  - Intervention  │         │  │  - Demos           │  │   │
    │  │  - 50Hz control  │         │  │  - Online data     │  │   │
    │  │  - Log to buffer │         │  │  - Interventions   │  │   │
    │  └──────────────────┘         │  └────────────────────┘  │   │
    │                                │            │              │   │
    │                                │            ▼              │   │
    │                                │  ┌────────────────────┐  │   │
    │                                │  │  World Model       │  │   │
    │                                │  │  Update            │  │   │
    │                                │  └────────┬───────────┘  │   │
    │                                │           │              │   │
    │                                │           ▼              │   │
    │                                │  ┌────────────────────┐  │   │
    │                                │  │  Imagination       │  │   │
    │                                │  │  Rollouts (10+)    │  │   │
    │                                │  └────────┬───────────┘  │   │
    │                                │           │              │   │
    │                                │           ▼              │   │
    │                                │  ┌────────────────────┐  │   │
    │                                │  │  Imagination       │  │   │
    │                                │  │  Buffer            │  │   │
    │                                │  └────────┬───────────┘  │   │
    │                                │           │              │   │
    │  ┌──────────────────┐         │           ▼              │   │
    │  │  Policy Update   │ ◄───────┼──  ┌────────────────┐   │   │
    │  │  (Synced every   │         │    │  RL Training   │   │   │
    │  │   20 seconds)    │         │    │  (SAC +  IL)   │   │   │
    │  └──────────────────┘         │    └────────────────┘   │   │
    │                                └───────────────────────────┘   │
    └────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Weights & Biases    │
                          │  Logging             │
                          └──────────────────────┘
```

---

## Training Pipeline

### Stage 1: Imitation Learning Pre-training

**Objective**: Initialize VLA policy with broad manipulation capabilities

```bash
python scripts/train_vla.py \
    --config configs/vla_training.yaml \
    --data_dir data/demonstrations \
    --output_dir checkpoints/vla_pretrained
```

**Duration**: Hours to days (depending on demo dataset size)

**Output**: `vla_pretrained.pth` (policy checkpoint)

---

### Stage 2: Reward Classifier Training

**Objective**: Train binary classifier to recognize task success

```bash
python scripts/train_reward.py \
    --config configs/reward_classifier.yaml \
    --demo_dir data/demonstrations \
    --output_dir checkpoints/reward_classifier
```

**Duration**: 30-60 minutes

**Output**: `reward_classifier.pth`

---

### Stage 3: Integrated Training

**Objective**: Online RL with world model imagination

```bash
python scripts/train_integrated.py \
    --config configs/integrated_training.yaml \
    --vla_checkpoint checkpoints/vla_pretrained.pth \
    --reward_checkpoint checkpoints/reward_classifier.pth \
    --output_dir checkpoints/integrated
```

**Training Loop** (pseudocode):

```python
# Initialize
policy = load_pretrained_vla()
world_model = DreamerV3()
replay_buffer = ReplayBuffer()
imagination_buffer = ImaginationBuffer()

# Load demos
replay_buffer.add_demonstrations(demo_data)

# Asynchronous training
async def robot_collection_loop():
    while training:
        obs = env.reset()
        done = False

        while not done:
            # Policy inference
            action = policy(obs, instruction)

            # Check for human intervention
            if intervention_detected():
                action = get_human_command()
                intervention_flag = True

            # Execute on robot
            next_obs, reward, done = env.step(action)

            # Log to replay buffer
            replay_buffer.add(obs, action, reward, done, intervention_flag)

            obs = next_obs

        # Sync policy every 20 seconds
        if time_to_sync():
            policy.load_state_dict(latest_policy)

async def training_loop():
    while training:
        # 1. Update world model
        real_batch = replay_buffer.sample(batch_size=256)
        world_model_loss = world_model.train_step(real_batch)

        # 2. Generate imagined rollouts
        for _ in range(num_imagination_rollouts):
            initial_state = replay_buffer.sample_state()
            imagined_trajectory = world_model.imagine(
                initial_state,
                policy,
                horizon=imagination_horizon
            )
            imagination_buffer.add(imagined_trajectory)

        # 3. Train policy on mixed data
        real_batch = replay_buffer.sample(batch_size=128, include_demos=True)
        imagined_batch = imagination_buffer.sample(batch_size=128)
        mixed_batch = concatenate(real_batch, imagined_batch)

        # Compute hybrid loss
        rl_loss = compute_sac_loss(mixed_batch)
        il_loss = compute_bc_loss(demo_batch)
        total_loss = rl_loss + lambda_decay(step) * il_loss

        # Update policy
        policy.optimizer.step(total_loss)

        # Log metrics
        wandb.log({
            'world_model_loss': world_model_loss,
            'rl_loss': rl_loss,
            'il_loss': il_loss,
            'imagination_horizon': imagination_horizon,
            'intervention_rate': intervention_rate,
        })

# Run both loops concurrently
await asyncio.gather(
    robot_collection_loop(),
    training_loop()
)
```

**Duration**: 1-2 hours for convergence on single task

**Outputs**:
- `integrated_policy_{step}.pth` (checkpoints every N steps)
- Weights & Biases logs

---

### Stage 4: Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/integrated_final.pth \
    --num_episodes 50 \
    --tasks "pick_and_place,fold_towel"
```

**Metrics**:
- Success rate (%)
- Task completion time (seconds)
- Intervention count (should be 0 in evaluation)
- Generalization to unseen object configurations

---

## Hardware Architecture

### Compute Resources

```
┌──────────────────────────────────────────────────────────────┐
│                    WORKSTATION                               │
│                                                              │
│  CPU: High-core-count (allocate separate cores for robot    │
│       collection vs training)                               │
│                                                              │
│  GPUs: 8× NVIDIA H100 (80GB each)                           │
│  ├─ GPU 0-1: Policy inference (robot collection)            │
│  ├─ GPU 2-5: RL training + world model training             │
│  └─ GPU 6-7: Imagination rollouts (parallel simulation)     │
│                                                              │
│  RAM: 256GB+ (for replay buffers and data loading)          │
│  Storage: NVMe SSD for fast data I/O                        │
└──────────────────────────────────────────────────────────────┘
            │
            │ USB/Serial
            ▼
┌──────────────────────────────────────────────────────────────┐
│                    ROBOT HARDWARE                            │
│                                                              │
│  ┌────────────────┐        ┌────────────────┐              │
│  │  ARX X5 (Left) │        │  ARX X5 (Right)│              │
│  │  - 6 DoF       │        │  - 6 DoF       │              │
│  │  - Gripper     │        │  - Gripper     │              │
│  │  - 50Hz control│        │  - 50Hz control│              │
│  └────────────────┘        └────────────────┘              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Cameras (Fish-eye)                                   │  │
│  │  ├─ Left camera                                       │  │
│  │  ├─ Right camera                                      │  │
│  │  └─ Base camera                                       │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Software Stack

### Core Dependencies

| Component | Framework | Version | Purpose |
|-----------|-----------|---------|---------|
| VLA (π0.5) | PyTorch | 2.2+ | Policy inference |
| World Model | JAX | 0.4.35+ | Fast simulation |
| RL (HIL-SERL) | JAX/Flax | 0.4.35+ | Training efficiency |
| Data Format | LeRobot | Latest | RLDS datasets |
| Robot Control | ARX X5 SDK | Latest | Hardware interface |
| Logging | Weights & Biases | Latest | Experiment tracking |

### Directory Structure

```
IL_RL_WorldModel/
├── models/
│   ├── vla/
│   │   ├── pi05_policy.py          # π0.5 VLA integration
│   │   ├── vision_encoder.py       # SigLIP ViT
│   │   ├── language_encoder.py     # Gemma 2B
│   │   └── action_decoder.py       # Diffusion decoder
│   ├── world_model/
│   │   ├── dreamer_v3.py           # RSSM world model
│   │   ├── encoder.py              # Vision encoder for WM
│   │   ├── dynamics.py             # GRU dynamics model
│   │   └── decoders.py             # Observation/reward decoders
│   └── reward/
│       └── reward_classifier.py    # Binary classifier
├── training/
│   ├── imitation/
│   │   └── behavioral_cloning.py   # VLA pre-training
│   ├── rl/
│   │   ├── sac_trainer.py          # Soft Actor-Critic
│   │   ├── replay_buffer.py        # Experience storage
│   │   └── hybrid_loss.py          # RL + IL loss
│   └── world_model/
│       └── wm_trainer.py           # World model training
├── robot/
│   ├── arx_x5/
│   │   ├── controller.py           # Robot interface
│   │   ├── camera.py               # Camera interface
│   │   └── safety.py               # Safety checks
│   └── teleop/
│       └── keyboard_intervention.py # Human control
├── data/
│   ├── demonstrations/             # RLDS demos
│   ├── replay_buffer/              # Online experiences
│   └── imagination/                # World model rollouts
├── evaluation/
│   └── metrics/
│       ├── success_rate.py
│       └── timing.py
├── configs/
│   ├── vla_training.yaml
│   ├── reward_classifier.yaml
│   ├── integrated_training.yaml
│   └── robot_config.yaml
├── scripts/
│   ├── train_vla.py
│   ├── train_reward.py
│   ├── train_integrated.py
│   └── evaluate.py
└── tests/
    ├── test_vla.py
    ├── test_world_model.py
    ├── test_rl_training.py
    └── test_robot_interface.py
```

---

## Component Interactions

### Initialization Phase

```
[Demo Data] ──┬──► [VLA Pre-training] ──► VLA Checkpoint
              │
              └──► [Reward Classifier Training] ──► Reward Checkpoint
```

### Training Phase

```
                    ┌─────────────────────┐
                    │   Policy (VLA init) │
                    └──────────┬──────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────┐
    │             Asynchronous Loop                     │
    │                                                   │
    │  Robot Collection     GPU Training                │
    │  ─────────────────    ────────────                │
    │  Execute policy   ──► Replay Buffer               │
    │  Log experiences      │                           │
    │  Human intervene      ▼                           │
    │                   World Model ──► Imagination     │
    │                       │           Buffer          │
    │                       │              │            │
    │                       ▼              ▼            │
    │  Policy sync  ◄─── RL Training (real + imagined) │
    │                                                   │
    └───────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Trained Policy       │
                    └──────────────────────┘
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
- [x] Repository setup and documentation
- [x] Exploration of all codebases
- [ ] ARX X5 robot interface implementation
- [ ] Data pipeline (RLDS format)
- [ ] Basic testing framework

### Phase 2: VLA Integration (Week 3-4)
- [ ] π0.5 model integration
- [ ] Pre-trained weight loading
- [ ] ARX X5 action space adaptation
- [ ] VLA pre-training script
- [ ] Reward classifier implementation

### Phase 3: World Model (Week 5-6)
- [ ] DreamerV3 integration
- [ ] Language-conditioned dynamics
- [ ] Imagination rollout mechanism
- [ ] World model training script

### Phase 4: RL Pipeline (Week 7-8)
- [ ] SAC implementation with demos
- [ ] Hybrid loss (RL + IL)
- [ ] Replay buffer with intervention tracking
- [ ] Human intervention system (keyboard)

### Phase 5: Integration (Week 9-10)
- [ ] Asynchronous training loop
- [ ] Multi-GPU distributed training
- [ ] Weights & Biases integration
- [ ] End-to-end testing

### Phase 6: Evaluation (Week 11-12)
- [ ] Evaluation scripts and metrics
- [ ] Baseline comparisons
- [ ] Ablation studies
- [ ] Performance benchmarking

### Phase 7: Scaling (Week 13+)
- [ ] Multi-task training
- [ ] Curriculum learning
- [ ] Zero-shot generalization tests
- [ ] Publication preparation

---

## Next Steps

**Immediate priorities** (in order):

1. **ARX X5 Interface**: Implement `robot/arx_x5/controller.py`
   - 50Hz control loop
   - Proprioception reading
   - Safety checks

2. **Data Pipeline**: Implement `data/rlds_loader.py`
   - LeRobot dataset loading
   - RLDS format conversion
   - Data augmentation

3. **VLA Integration**: Implement `models/vla/pi05_policy.py`
   - Load OpenPI pretrained weights
   - Adapt for ARX X5 (14D action space)
   - Inference at 50Hz

4. **Reward Classifier**: Implement `models/reward/reward_classifier.py`
   - ResNet-10 encoder
   - Binary classification
   - Training from demos

5. **World Model**: Implement `models/world_model/dreamer_v3.py`
   - RSSM architecture
   - Language conditioning
   - Imagination rollouts

6. **RL Training**: Implement `training/rl/sac_trainer.py`
   - SAC with demonstrations
   - Hybrid loss
   - Intervention handling

7. **Async Training**: Implement `scripts/train_integrated.py`
   - Concurrent robot collection and training
   - Policy synchronization
   - Real-time logging

---

*Last updated: October 2025*
*This architecture document is a living document and will be updated as the project evolves.*
