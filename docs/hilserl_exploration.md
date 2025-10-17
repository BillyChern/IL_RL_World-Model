# HIL-SERL (Human-in-the-Loop SERL) Exploration

## Overview

HIL-SERL is a vision-based reinforcement learning system for robotic manipulation that combines human guidance with autonomous learning. Developed by the RAIL (Robot AI & Learning Lab) at UC Berkeley, it achieves near-perfect success rates on precise, dexterous manipulation tasks within 1-2.5 hours of training.

**Repository**: https://github.com/rail-berkeley/hil-serl
**Project Website**: https://hil-serl.github.io/
**Paper**: "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning" (arXiv:2410.21845)
**Authors**: Jianlan Luo, Charles Xu, Jeffrey Wu, Sergey Levine (2024)
**License**: Apache 2.0

### Key Results
- **2x improvement** in success rate over imitation learning baselines
- **1.8x faster execution** compared to prior RL approaches
- Training times: 1-2.5 hours for complex real-world tasks
- Handles diverse tasks: dynamic manipulation, precision assembly, dual-arm coordination

---

## Repository Structure

```
hil-serl/
├── serl_launcher/                  # Core HIL-SERL implementation
│   ├── agents/                     # Policy implementations (SAC, BC)
│   │   └── continuous/            # Continuous action space agents
│   │       ├── bc.py              # Behavioral Cloning
│   │       ├── sac.py             # Soft Actor-Critic
│   │       ├── sac_hybrid_dual.py # Hybrid SAC (dual config)
│   │       └── sac_hybrid_single.py # Hybrid SAC (single config)
│   ├── common/                    # Shared utilities
│   ├── data/                      # Data handling and management
│   │   ├── data_store.py          # Thread-safe data collection interface
│   │   ├── dataset.py             # Dataset utilities
│   │   ├── replay_buffer.py       # Standard replay buffer
│   │   └── memory_efficient_replay_buffer.py # Optimized replay buffer
│   ├── networks/                  # Neural network architectures
│   │   ├── actor_critic_nets.py   # Actor-critic architectures
│   │   ├── classifier.py          # Classification networks
│   │   ├── lagrange.py            # Lagrange multipliers
│   │   ├── mlp.py                 # Multi-layer perceptrons
│   │   └── reward_classifier.py   # Binary reward classifier
│   ├── utils/                     # Utility functions
│   ├── vision/                    # Computer vision components
│   └── wrappers/                  # Environment wrappers
│       ├── chunking.py            # Data segmentation
│       ├── norm.py                # Normalization
│       ├── remap.py               # Action remapping
│       ├── serl_obs_wrappers.py   # Observation processing
│       ├── video_recorder.py      # Video recording
│       └── video_wrapper.py       # Video wrapper utilities
├── serl_robot_infra/              # Hardware integration layer
│   ├── franka_env/                # Franka robot Gym environment
│   │   ├── camera/                # Camera functionality
│   │   ├── envs/                  # Environment implementations
│   │   │   ├── franka_env.py      # Base Franka environment
│   │   │   ├── dual_franka_env.py # Dual-arm environment
│   │   │   ├── franka_wrench_env.py # Wrench-based control
│   │   │   ├── relative_env.py    # Relative action space
│   │   │   └── wrappers.py        # Environment wrappers
│   │   ├── spacemouse/            # SpaceMouse controller integration
│   │   │   ├── pyspacemouse.py    # SpaceMouse driver
│   │   │   ├── spacemouse_expert.py # Human intervention interface
│   │   │   └── spacemouse_test.py # Testing utilities
│   │   └── utils/                 # Utility functions
│   ├── robot_servers/             # Flask-based ROS command interface
│   └── egg_flip_controller/       # Wrench controller for dynamic tasks
├── examples/                      # Training scripts and examples
│   ├── train_bc.py                # Behavioral cloning training
│   ├── train_hgdagger.py          # HG-DAgger training
│   ├── train_rlpd.py              # RLPD training (main HIL-SERL)
│   ├── train_reward_classifier.py # Reward classifier training
│   ├── record_demos.py            # Demonstration collection
│   ├── record_success_fail.py     # Success/failure trajectory collection
│   └── experiments/               # Task-specific configurations
└── docs/                          # Documentation
    ├── franka_walkthrough.md      # Franka setup guide
    └── images/                    # Documentation images
```

---

## Key Components and Architecture

### 1. Distributed Actor-Learner Architecture

HIL-SERL uses an asynchronous actor-learner pattern enabled by the **Agentlace** framework for network communication.

**Agentlace Framework**:
- Distributed ML framework for agent policies
- Three primary modes: Action, Inference, Trainer
- Supports async communication between different processes/machines
- Learner periodically synchronizes policies with distributed actor nodes

**Architecture Flow**:
```
┌─────────────────┐         ┌─────────────────┐
│  Actor Process  │         │ Learner Process │
│  (Robot/Env)    │◄────────┤  (GPU Training) │
│                 │  Policy │                 │
│  - Collects data│  Update │  - Trains agent │
│  - Executes     │         │  - Updates nets │
│  - Interventions│────────►│  - Logs metrics │
└─────────────────┘  Buffer └─────────────────┘
                     Data
```

---

### 2. Human Intervention Mechanism

#### SpaceMouse Integration

**Hardware**: 3Dconnexion SpaceMouse (3D input device)
- 6 DOF control: x, y, z translation + roll, pitch, yaw rotation
- Supports dual SpaceMouse setup (12 DOF for bimanual tasks)
- Real-time button state tracking

**Implementation** (`spacemouse_expert.py`):
```python
# Multiprocessing-based continuous input capture
- Daemon process continuously reads SpaceMouse state
- Coordinate transformation: negates y-axis and roll values
- Returns: np.array(action) + button_states
- Dual device: concatenates outputs for 12 DOF
```

**Intervention Detection**:
- Button presses signal human takeover
- System tracks:
  - `intervention_count`: Number of episodes with interventions
  - `intervention_steps`: Total intervention actions taken
  - Separate `intvn_data_store` for intervened transitions

#### Intervention Workflow

1. **Real-time Monitoring**: Actor process monitors SpaceMouse button states during execution
2. **Takeover Signal**: When human presses button, `info['intervene_action']` is populated
3. **Action Override**: Human-provided action replaces agent action
4. **Data Storage**: Intervened transition stored in separate intervention buffer
5. **Training Integration**: Both autonomous and intervened data used for learning

**In Training Loop** (`train_rlpd.py`):
```python
# Check for intervention during environment step
if 'intervene_action' in info:
    actions = info.pop('intervene_action')  # Use human action
    intervention_count += 1  # Track intervention
    # Store in separate intervention buffer
    intvn_data_store.insert(transition)
```

---

### 3. Replay Buffer Architecture

#### Standard Replay Buffer (`replay_buffer.py`)

**Data Storage**:
- Pre-allocated NumPy arrays for fixed memory footprint
- Circular buffer: `_insert_index = (_insert_index + 1) % capacity`
- Stores: observations, actions, next_observations, rewards, masks, dones

**Optional Fields**:
- `next_actions`: For n-step returns
- `intervention_flags`: Boolean marking human interventions
- `labels`: Integer labels for demonstration classification
- `grasp_penalties`: Task-specific penalties

**Structure**:
```python
{
    'observations': dict (nested, handles images + state),
    'actions': np.array(capacity, action_dim),
    'next_observations': dict,
    'rewards': np.array(capacity,) float32,
    'masks': np.array(capacity,) float32,
    'dones': np.array(capacity,) bool,
    'labels': np.array(capacity,) int (optional),
}
```

**Sampling**:
- Inherits from `Dataset` class
- `get_iterator()`: Creates prefetch queue with JAX device placement
- GPU optimization for faster batch transfers

#### Data Store Wrapper (`data_store.py`)

**Purpose**: Thread-safe wrapper around replay buffers

**Classes**:
- `ReplayBufferDataStore`: Wraps standard replay buffer
- `MemoryEfficientReplayBufferDataStore`: Wraps optimized version

**Features**:
- Locking mechanisms for concurrent access
- `insert()` and `sample()` with thread safety
- `latest_data_id()`: Returns current insert index
- Utility functions for loading demonstrations

**Demonstration Loading**:
```python
# Populate demonstrations into data store
def populate_data_store_with_z_axis_only():
    # Filters state: removes x/y, keeps z and other dims
    # Enables task-specific state modifications
```

---

### 4. RL Algorithm: Soft Actor-Critic (SAC)

#### SAC Agent Implementation (`sac.py`)

**Core Components**:
1. **Critic Network**: Evaluates state-action pairs with ensemble
2. **Policy Network**: Generates action distributions (Gaussian)
3. **Temperature Parameter**: Entropy regularization via Lagrange multiplier
4. **Target Networks**: Soft-updated for training stability

**Network Architecture**:
- Ensemble critics: Typically 2 networks (configurable)
- Support for REDQ (Random Ensemble Q-learning)
- Support for TD3 variant
- Modular forward passes: `forward_critic()`, `forward_policy()`, `forward_temperature()`

**Training Loop** (`update()` method):
1. Unpack batch data (observations, actions, rewards, next_obs, masks)
2. Apply data augmentation (e.g., random crop for images)
3. Add reward bias (for demonstration weighting)
4. Compute three losses:
   - **Critic Loss**: MSE between predicted and target Q-values
   - **Policy Loss**: Entropy-regularized actor objective
   - **Temperature Loss**: Lagrange penalty for entropy constraint
5. Update target networks via soft updates

**Hyperparameters** (defaults):
```python
discount = 0.95
soft_target_update_rate = 0.005  # τ for target nets
target_entropy = -action_dim / 2  # Automatic entropy tuning
critic_ensemble_size = 2
learning_rate = 3e-4  # All networks
backup_entropy = True  # Include entropy in target
```

**Batch Format**:
```python
{
    "observations": nested dict,
    "actions": np.array,
    "next_observations": nested dict,
    "rewards": np.array,
    "masks": np.array,  # 1 - done
}
```

#### Behavioral Cloning Agent (`bc.py`)

**Purpose**: Pre-training or baseline comparison

**Loss Function**:
```python
actor_loss = -(log_probs).mean()  # Negative log-likelihood
mse = ((pi_actions - batch_actions) ** 2).sum(-1)  # Diagnostic
```

**Features**:
- Supervised learning on expert demonstrations
- Data augmentation: random crop (padding=4) on images
- Supports tanh-squashed and unsquashed action distributions
- Clips expert actions to [-1+1e-6, 1-1e-6] for squashed outputs

**Relationship to SAC**:
- Separate implementation (pure imitation)
- Can serve as initialization for SAC policy
- Used as baseline for comparison

---

### 5. Reward Classifier

#### Purpose
Binary classifier trained on success/failure demonstrations to provide sparse reward signals during RL training.

#### Architecture (`reward_classifier.py`)

**Network Structure**:
```python
BinaryClassifier:
  ├── PreTrainedResNetEncoder (ResNet-10, ImageNet-1K)
  │   └── EncodingWrapper
  │       └── Spatial learned embeddings (8 blocks, 256-dim)
  ├── Dense layer (256 hidden units)
  ├── Dropout (0.1)
  ├── Layer Normalization
  ├── ReLU activation
  └── Output layer (1 logit for binary classification)

NWayClassifier:
  └── Same structure, but outputs n_way logits
```

**Input Format**:
- Observation dictionary containing image(s)
- Multiple image keys supported (separate encoders per key)
- Outputs combined via learned spatial pooling

#### Training Procedure (`train_reward_classifier.py`)

**Data Collection**:
1. Teleoperator records success and failure trajectories using `record_success_fail.py`
2. Positive buffer: Successful trajectories (label=1)
3. Negative buffer: Failure trajectories (label=0)

**Training Loop** (default: 150 epochs):
1. Sample equal batches from positive and negative buffers
2. Apply data augmentation: batched random cropping (padding=4)
3. Compute loss: Sigmoid binary cross-entropy
4. Optimize with Adam (learning_rate=1e-4)
5. Evaluate: Accuracy via sigmoid-threshold (≥0.5) vs. labels

**Usage in RL**:
- Provides sparse binary reward: 1.0 for success, 0.0 for failure
- JIT-compiled for fast inference during training
- Checkpoint saved after training

---

### 6. Training Pipeline: RLPD

#### Overview
RLPD (Reinforcement Learning from Preferences and Demonstrations) is the main training script implementing HIL-SERL.

#### Training Stages

**Stage 1: Data Collection**
1. **Record Success/Failure Examples** (`record_success_fail.py`):
   - Teleoperator uses SpaceMouse to perform task multiple times
   - Label trajectories as success (1) or failure (0)
   - Store in separate pickle files

2. **Train Reward Classifier** (`train_reward_classifier.py`):
   - Train binary classifier on labeled data
   - Provides reward signal for RL

3. **Collect Demonstrations** (`record_demos.py`):
   - Teleoperator performs successful task executions
   - Store in demonstration buffer
   - Used for 50/50 sampling with online data

**Stage 2: RL Training** (`train_rlpd.py`)

**Actor Process** (runs on robot/environment):
```python
while True:
    # Get action from current policy
    action = agent.sample_actions(observation)

    # Check for human intervention
    next_obs, reward, done, info = env.step(action)
    if 'intervene_action' in info:
        action = info.pop('intervene_action')
        intervention_count += 1
        # Store in intervention buffer
        intvn_data_store.insert(transition)
    else:
        # Store in regular replay buffer
        replay_buffer.insert(transition)

    # Periodically save buffer to disk
    if step % save_interval == 0:
        replay_buffer.save()
```

**Learner Process** (runs on GPU):
```python
# Wait for sufficient data
while replay_buffer.size < training_starts:
    time.sleep(1)

while True:
    # Sample from both replay and demo buffers (50/50)
    online_batch = replay_buffer.sample(batch_size // 2)
    demo_batch = demo_buffer.sample(batch_size // 2)
    batch = concatenate(online_batch, demo_batch)

    # Perform critic-only updates (n-1 times)
    for _ in range(num_critic_updates - 1):
        agent.update(batch, update_actor=False)

    # Full network update (critic + actor + temperature)
    agent.update(batch, update_actor=True)

    # Periodically publish updated parameters to actor
    if step % sync_interval == 0:
        trainer_client.publish(agent.state)

    # Log metrics
    wandb.log(metrics)
```

**Key Features**:
- **50/50 Sampling**: Half from online replay, half from demonstrations
- **Multiple Critic Updates**: n-1 critic-only + 1 full update per iteration
- **Asynchronous**: Actor collects while learner trains
- **Human-in-the-Loop**: Interventions seamlessly integrated
- **Distributed**: Actor and learner can run on separate machines

---

### 7. Environment and Observation Handling

#### SERL Observation Wrapper (`serl_obs_wrappers.py`)

**Purpose**: Restructure observations into state-image format

**Functionality**:
- Accepts `proprio_keys` to specify which state components to include
- Flattens proprioceptive state into continuous vector
- Preserves image observations separately
- Returns: `{"state": flattened_state, **images}`

**Implementation**:
```python
def observation(obs):
    state = flatten_state(obs, proprio_keys)
    images = obs["images"]
    return {"state": state, **images}
```

#### Franka Environment (`franka_env.py`)

**Base Class**: `gymnasium.Env`

**Features**:
- Flask server communication via HTTP POST requests
- ROS integration for robot control
- Real-time camera feeds (threaded display)
- Keyboard termination (ESC key listener)
- Configurable action space (absolute, relative, wrench-based)

**Action Control Modes**:
- `franka_env.py`: Standard impedance control
- `relative_env.py`: Relative action space (delta pose)
- `franka_wrench_env.py`: Wrench-based control for dynamic tasks
- `dual_franka_env.py`: Bimanual coordination

**Communication Architecture**:
```
Gym Environment (Python)
    ↓ HTTP POST
Flask Server
    ↓ ROS Messages
Robot Controllers
    ↓ Joint Commands
Franka Robot
```

---

### 8. Vision and Neural Networks

#### Network Architectures

**Actor-Critic Networks** (`actor_critic_nets.py`):
- Modular design for policy and value functions
- Support for image observations via CNN encoders
- Concatenation of proprioceptive state + visual features

**MLP Networks** (`mlp.py`):
- Multi-layer perceptrons for function approximation
- Configurable hidden layers and activations

**Lagrange Networks** (`lagrange.py`):
- Lagrange multipliers for constrained optimization
- Used for automatic entropy tuning in SAC

#### Vision Models

**Pretrained Encoder**:
- ResNet-10 backbone (pretrained on ImageNet-1K)
- Spatial learned embeddings pooling (8 blocks, 256-dim)
- Frozen or fine-tuned depending on task

**Data Augmentation**:
- Random crop with padding (typically padding=4)
- Applied during training for robustness
- Batched augmentation for efficiency

---

## Integration Approach for VLA + RL + World Model

### Mapping HIL-SERL to Project Architecture

#### Current HIL-SERL Components → Project Components

| HIL-SERL Component | Project Equivalent | Integration Notes |
|--------------------|-------------------|-------------------|
| **SAC Agent** | RL Training Pipeline | Use as RL backbone, modify for VLA initialization |
| **Behavioral Cloning** | VLA Pre-training | Initialize VLA via IL on heterogeneous data |
| **Reward Classifier** | Reward Function Module | Binary classifier for success detection |
| **Replay Buffer** | Real Data Buffer | Store real robot experiences |
| **Demo Buffer** | Demonstration Module | Bootstrap RL with expert data |
| **Intervention System** | Human-in-the-Loop Protocol | Safety and correction mechanism |
| **Agentlace** | Distributed Training Framework | Actor-learner separation |
| **Franka Environment** | Robot Interface | Standardize via Gym API |

#### Proposed Extensions

**1. World Model Integration**

Add parallel world model training to existing architecture:

```python
# New components to add:
world_model/
├── dreamer.py              # DreamerV3-style latent dynamics
├── video_predictor.py      # Diffusion/JEPA for next frame prediction
└── imagination_buffer.py   # Store imagined rollouts

# Modified training loop:
while True:
    # Existing: Actor collects real data
    real_batch = replay_buffer.sample(batch_size // 2)

    # NEW: World model generates imagined data
    imagined_batch = imagination_buffer.sample(batch_size // 2)

    # Existing: Demos
    demo_batch = demo_buffer.sample(batch_size // 4)

    # Train on mixture
    batch = concatenate(real_batch, imagined_batch, demo_batch)
    agent.update(batch)
```

**Mixing Ratios**:
- Start: 60% real, 20% imagined, 20% demos
- Mid-training: 40% real, 40% imagined, 20% demos
- Late-training: 30% real, 50% imagined, 20% demos

**2. VLA Policy Initialization**

Replace BC pre-training with VLA:

```python
# Current: BC agent trains on demos
bc_agent = BCAgent(...)
bc_agent.train(demo_buffer)

# NEW: Initialize with pre-trained VLA
vla_policy = load_pretrained_vla("pi0.5_weights")
sac_agent = SACAgent(
    policy_network=vla_policy,  # Initialize with VLA
    freeze_encoder=True,  # Freeze vision/language encoders
)
```

**3. Language Conditioning**

Extend observations to include language instructions:

```python
# Modified observation wrapper
class VLAObsWrapper:
    def observation(obs):
        state = flatten_state(obs)
        images = obs["images"]
        instruction_embedding = encode_instruction(obs["instruction"])
        return {
            "state": state,
            "instruction": instruction_embedding,
            **images
        }

# Modified networks to accept instruction
class VLAPolicy:
    def forward(obs):
        state = obs["state"]
        images = obs["images"]
        instruction = obs["instruction"]

        visual_features = vision_encoder(images)
        language_features = language_encoder(instruction)
        combined = concatenate([state, visual_features, language_features])

        return policy_head(combined)
```

**4. Multi-Task Training**

Extend environment to support multiple tasks:

```python
# Task-specific environments
task_envs = {
    "pick_and_place": PickAndPlaceEnv(),
    "cable_routing": CableRoutingEnv(),
    "assembly": AssemblyEnv(),
}

# Sample tasks during training
while True:
    task = sample_task()
    instruction = task_instructions[task]
    env = task_envs[task]

    obs = env.reset()
    obs["instruction"] = instruction

    # Standard RL loop with language-conditioned policy
    action = agent.sample_actions(obs)
    ...
```

---

## Dependencies

### Core Framework Dependencies

**Deep Learning**:
- `jax >= 0.4.35` (with CUDA 12 for GPU)
- `flax >= 0.8.0` (neural network library)
- `optax >= 0.1.5` (optimization)
- `chex >= 0.1.85` (JAX utilities)

**Reinforcement Learning**:
- `gymnasium == 0.29.1` (environment API)
- `distrax >= 0.1.2` (probability distributions)

**Robotics**:
- ROS (Robot Operating System) for Franka control
- Flask for server-client communication

**Computer Vision**:
- `tensorflow >= 2.15.0`
- `tensorflow_probability >= 0.23.0`
- `einops >= 0.6.1` (tensor operations)
- `imageio >= 2.31.1`
- `moviepy >= 1.0.3`

**Utilities**:
- `numpy >= 1.24.3`
- `scipy == 1.11.4`
- `ml_collections >= 0.1.0` (config management)
- `tqdm >= 4.60.0` (progress bars)
- `wandb >= 0.12.14` (experiment tracking)
- `absl-py >= 0.12.0` (flags and logging)

**Other**:
- `agentlace` (distributed actor-learner framework)
- `pynput` (keyboard/mouse input)
- `natsort` (natural sorting)
- `matplotlib` (plotting)
- `pre-commit == 3.3.3` (code quality)

### Hardware Requirements

**Robot**:
- Franka Emika Panda arm(s)
- 3Dconnexion SpaceMouse (single or dual)
- RGB camera(s) (USB or ROS-compatible)

**Compute**:
- GPU with CUDA 12 support (for JAX)
- Multi-core CPU for parallel data collection
- Network connectivity for distributed training

---

## Installation Summary

```bash
# 1. Create conda environment
conda create -n hilserl python=3.10
conda activate hilserl

# 2. Install JAX with GPU support
pip install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Install SERL launcher
cd serl_launcher
pip install -e .
pip install -r requirements.txt

# 4. Install robot infrastructure (if using real robot)
cd ../serl_robot_infra
pip install -e .
# Follow additional ROS setup instructions

# 5. Install agentlace
pip install agentlace
```

---

## Training Workflow Summary

### Complete Workflow

1. **Setup Environment**:
   - Configure Franka robot and camera
   - Test SpaceMouse input
   - Calibrate workspace

2. **Collect Success/Failure Examples**:
   ```bash
   python examples/record_success_fail.py --config experiments/task_config.py
   ```
   - Record 20-50 success trajectories
   - Record 20-50 failure trajectories

3. **Train Reward Classifier**:
   ```bash
   python examples/train_reward_classifier.py --config experiments/task_config.py
   ```
   - Trains binary classifier on collected data
   - Provides sparse reward signal for RL

4. **Collect Demonstrations**:
   ```bash
   python examples/record_demos.py --config experiments/task_config.py
   ```
   - Record 10-20 expert demonstrations
   - Used for 50/50 sampling during RL

5. **Train RL Policy with HIL**:
   ```bash
   # Start learner process (on GPU machine)
   python examples/train_rlpd.py --learner --config experiments/task_config.py

   # Start actor process (on robot machine)
   python examples/train_rlpd.py --actor --config experiments/task_config.py
   ```
   - Actor collects data with interventions
   - Learner trains on real + demo data
   - Human intervenes when needed (via SpaceMouse)

6. **Evaluation**:
   - Disable interventions
   - Run policy autonomously
   - Measure success rate and cycle time

### Key Training Parameters

```python
# RL Hyperparameters
batch_size = 256
training_starts = 1000  # Min samples before training
discount = 0.95
learning_rate = 3e-4
target_entropy = -action_dim / 2

# Sampling
replay_ratio = 0.5  # 50% online, 50% demo
num_critic_updates = 1  # Critic-only updates per step
update_frequency = 1  # Update every step

# Distributed
sync_interval = 100  # Steps between policy syncs
save_interval = 1000  # Steps between buffer saves

# Intervention
allow_interventions = True
intervention_window = "anytime"  # When human can intervene
```

---

## Key Insights for Integration

### 1. Intervention System is Elegant
- Simple button press + action override
- Separate tracking for intervention data
- No complex state machines or mode switching
- Can be adapted for keyboard/gamepad inputs

### 2. 50/50 Demo Sampling is Critical
- Prevents catastrophic forgetting
- Maintains behavioral prior
- Compatible with VLA initialization

### 3. Distributed Architecture Enables Scaling
- Agentlace framework is lightweight
- Easy to add multiple actors
- Easy to add world model as separate process

### 4. Vision-Based Learning is Standard
- ResNet-10 encoder (pretrained)
- Random crop augmentation
- Directly compatible with VLA vision encoders

### 5. Reward Classifier Simplifies Reward Design
- Binary success/failure is sufficient
- Trained from human labels
- Can be extended to multi-class or dense rewards

### 6. Modular Design Enables Extensions
- Clear separation: agents, data, networks, environments
- Easy to swap SAC for other algorithms (PPO, TD3)
- Easy to add world model training loop

### 7. JAX/Flax Ecosystem
- Fast JIT compilation
- Efficient GPU utilization
- Compatible with modern RL libraries (Mujoco MJX, etc.)

---

## Recommended Integration Steps

### Phase 1: Replicate HIL-SERL Baseline
1. Set up HIL-SERL on simple manipulation task
2. Verify intervention system works
3. Train reward classifier and RL policy
4. Measure baseline performance

### Phase 2: Add VLA Pre-training
1. Implement VLA policy network (based on pi0.5)
2. Pre-train on heterogeneous dataset
3. Use as initialization for SAC
4. Compare to BC baseline

### Phase 3: Add World Model
1. Implement DreamerV3 or video prediction model
2. Train on replay buffer data
3. Generate imagined rollouts
4. Mix real + imagined data for policy training

### Phase 4: Language Conditioning
1. Extend observations to include instructions
2. Modify networks to accept language embeddings
3. Train on multiple tasks concurrently
4. Test zero-shot generalization

### Phase 5: Full Integration
1. Combine VLA + HIL-SERL + World Model
2. Train on diverse tasks
3. Evaluate sample efficiency gains
4. Measure generalization to new tasks

---

## References

**HIL-SERL**:
- Paper: https://arxiv.org/abs/2410.21845
- Code: https://github.com/rail-berkeley/hil-serl
- Website: https://hil-serl.github.io/

**Related Work**:
- SERL: https://github.com/rail-berkeley/serl
- Agentlace: https://github.com/youliangtan/agentlace
- DreamerV3: https://arxiv.org/abs/2301.04104
- π₀.₅ (pi0.5): Vision-Language-Action model

**Frameworks**:
- JAX: https://github.com/google/jax
- Flax: https://github.com/google/flax
- Gymnasium: https://gymnasium.farama.org/

---

## Notes

- HIL-SERL focuses on **precision and dexterity** rather than broad generalization
- Training times (1-2.5 hours) are for **specific tasks**, not multi-task policies
- Intervention system is **reactive** (human corrects mistakes) not proactive (human guides exploration)
- Reward classifier training is **offline** before RL, not online during RL
- Demo buffer is **static** during RL, not continuously updated
- Architecture is **vision-based** (no privileged simulation state)
- Framework is **hardware-agnostic** (can swap Franka for other robots)
- World model integration is **not implemented** but architecture supports it

---

**Document Created**: 2025-10-18
**Repository Version**: Latest (main branch)
**Status**: Complete exploration of HIL-SERL codebase
