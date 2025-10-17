# DreamerV3 and DayDreamer Exploration

## Overview

This document provides a comprehensive exploration of DreamerV3 and DayDreamer implementations, covering their architecture, training pipelines, and integration approaches for our IL/RL/WorldModel project.

**DreamerV3** is a general reinforcement learning algorithm that learns a world model from experiences and uses it to train actor-critic policies from imagined trajectories. It masters diverse domains with fixed hyperparameters.

**DayDreamer** applies DreamerV3 (specifically built on DreamerV2) to real-world physical robots, enabling training from small amounts of real-world experience without simulators through asynchronous actor-learner architecture.

---

## Repository Information

### DreamerV3 Official Repository
- **URL**: https://github.com/danijar/dreamerv3
- **Framework**: JAX (GPU/TPU/CPU support)
- **Language**: Python 3.11+
- **License**: MIT
- **Stars**: ~1,800

### DayDreamer Official Repository
- **URL**: https://github.com/danijar/daydreamer
- **Framework**: TensorFlow 2
- **Language**: Python
- **License**: MIT
- **Focus**: Real-world robot training

### Alternative Implementations
- **PyTorch**: https://github.com/NM512/dreamerv3-torch
- **PyTorch**: https://github.com/burchim/DreamerV3-PyTorch
- **Simplified PyTorch**: https://github.com/kc-ml2/SimpleDreamer

---

## World Model Architecture (RSSM)

### Recurrent State-Space Model (RSSM)

The RSSM is the core of DreamerV3's world model, combining stochastic latent states with deterministic recurrent history to model environmental uncertainty and enable long-horizon predictions.

#### Core Components

**1. Encoder** `qφ(zt | ht, xt)`
- Encodes image observations `xt` and recurrent hidden state `ht` into latent representation `zt`
- Architecture: CNN for images, MLP for vector inputs
- Output: Categorical distributions (32 classes × 32 categories by default)
- Depth: 64 with multipliers [2, 3, 4, 4]

**2. Sequence Model (Recurrent)**
- Predicts next hidden state from prior hidden state, latent representation, and action
- Architecture: Block GRU (Gated Recurrent Unit)
- Maintains deterministic state `ht` across time steps
- Default hidden size: 8192 dimensions

**3. Dynamics Predictor** `pφ(zt | ht)`
- Predicts latent representation from hidden state
- Trained to match encoder outputs (dynamics loss)
- Enables "dreaming" by predicting future states without observations

**4. Decoder** `pφ(x̂t | ht, zt)`
- Reconstructs observations from hidden state and latent representation
- Architecture: CNN for images (mirrors encoder), MLP for vectors
- Trained via prediction loss to reconstruct inputs

**5. Reward Predictor**
- Predicts immediate reward given `zt` and `ht`
- Architecture: MLP (1 layer, 1024 units by default)
- Output: Categorical distribution with 255 exponentially-spaced bins
- Uses symexp two-hot loss for robust prediction

**6. Continue Predictor**
- Predicts episode continuation flag (termination)
- Architecture: MLP
- Binary classification for episode end detection

### State Representation

The RSSM maintains two types of states:

- **Stochastic State** `zt`: Captured from observations, represents uncertainty
  - 32 one-hot vectors from 32 categorical distributions
  - Total: 32 classes × 32 categories = 1024 possible values
  - Uses straight-through gradients through sampling

- **Deterministic State** `ht`: Recurrent history
  - Maintained by Block GRU
  - Default dimension: 8192
  - Captures temporal dependencies

**Combined Feature** `feat = [ht, zt]` is used for all predictions

### Training Losses

1. **Prediction Loss**: Trains decoder, reward predictor, continue predictor
   - Reconstructs observations, predicts rewards and episode ends

2. **Dynamics Loss**: Trains dynamics predictor to fit encoder output
   - Ensures dynamics model can predict latent states

3. **Representation Loss**: Trains encoder to fit dynamics predictor output
   - Weight: 0.1 (balances with dynamics loss)
   - Uses KL divergence with free bits for categorical distributions

### Key Architectural Features

- **Block GRU**: More efficient than standard GRU
- **RMSNorm**: Root mean square normalization for stability
- **SiLu Activation**: Sigmoid Linear Unit (Swish)
- **Categorical Representations**: More expressive than Gaussian for world modeling
- **1% Unimix**: Adds uniform mixture to categorical distributions for exploration

---

## Imagination Mechanism

### Imagination Rollouts

DreamerV3 performs "dreaming" by using the learned world model to generate synthetic trajectories for policy training, dramatically reducing real environment interaction needs.

#### Process

1. **Start State**: Begin from a real state in the replay buffer `(ht, zt)`

2. **Action Selection**: Sample action from current policy `at ~ π(at | ht, zt)`

3. **Predict Next State**: Use dynamics model
   - Update hidden state: `ht+1 = GRU(ht, zt, at)`
   - Predict stochastic state: `zt+1 ~ pφ(zt+1 | ht+1)`

4. **Predict Reward**: `r̂t ~ reward_predictor(ht, zt)`

5. **Predict Continue**: `ct ~ continue_predictor(ht, zt)`

6. **Repeat**: Steps 2-5 for imagination horizon

7. **Return Trajectories**: `{(ht, zt, at, r̂t, ct)}` for policy optimization

#### Configuration

- **Imagination Horizon (imag_horizon)**: 15 steps (default)
  - Shorter horizons rely more on bootstrapping from value network
  - Longer horizons provide more accurate returns but risk model error accumulation
  - DreamerV3 uses λ-returns to be robust to horizon length

- **Batch Size**: 16 episodes
- **Batch Length**: 64 time steps per episode
- **Imagination Batch Size**: Up to 16,000 states for policy training

### Imagination vs Real Data

**Key Insight**: In DreamerV3/DayDreamer architecture:
- **Real data** → trains world model (supervised learning)
- **Imagined data** → trains actor-critic policy (RL)

This is NOT a mixing ratio in a single buffer, but a two-stage process:
1. World model learns from real replay buffer
2. Policy learns from imagined rollouts generated by world model

**Benefits**:
- Massively parallel policy optimization (16K+ imagined states per batch)
- No need to wait for real environment interactions
- Compact latent space enables efficient computation
- No image reconstruction needed during imagination

---

## Actor-Critic Algorithm

### Policy Network (Actor)

**Objective**: Maximize expected return while exploring

- **Architecture**: MLP with 1024 units × 3 layers
- **Output**: Action distribution
  - Discrete actions: Categorical distribution
  - Continuous actions: Tanh-transformed Gaussian (mean and std)
- **Training**: REINFORCE gradient estimator with entropy regularization
- **Entropy Weight**: Automatically adjusted to maintain target entropy
- **Gradients**: Do NOT backpropagate through world model

### Value Network (Critic)

**Objective**: Estimate expected returns for state-action pairs

- **Architecture**: MLP with 1024 units × 3 layers
- **Output**: Categorical distribution with 255 exponentially-spaced bins
  - Handles multi-modal return distributions
  - Robust to varying reward scales across domains
- **Training**: Regress λ-returns using symexp two-hot loss
- **Target Network**: Updated via exponential moving average (EMA)

### λ-Returns

DreamerV3 uses λ-returns to balance bias-variance tradeoff:

```
Gλ(st) = rt + γ[(1-λ)Vθ(st+1) + λGλ(st+1)]
```

- Interpolates between TD learning (λ=0) and Monte Carlo (λ=1)
- Default λ value provides good balance
- Makes algorithm robust to imagination horizon choice
- **Key Change from DreamerV2**: Uses current critic values, not target critic

### Training Process

1. Sample batch of real sequences from replay buffer
2. Encode with world model to get latent states
3. Perform imagination rollouts for each starting state
4. Compute λ-returns from imagined trajectories
5. Update actor to maximize returns (policy gradient)
6. Update critic to predict returns (regression)
7. Update world model on real data (supervised learning)

---

## Training Pipeline

### Configuration System

All configuration is in `dreamerv3/configs.yaml`:

```yaml
# Core training parameters
batch_size: 16              # Episodes per batch
batch_length: 64            # Time steps per episode
train_ratio: 32.0           # Gradient steps per env step
imag_horizon: 15            # Imagination rollout length
horizon: 333                # Episode length for λ-returns

# Model sizes (scale everything with model size)
deter: 8192                 # Deterministic state dimension
stoch: 32                   # Number of categorical distributions
classes: 64                 # Classes per categorical
hidden: 1024                # MLP hidden layer size

# Learning
learning_rate: 4e-5
warmup_steps: 1000
weight_decay: 0.0
beta1: 0.9
beta2: 0.999

# Logging
log_every: 120
report_every: 300
save_every: 900
```

**Model Size Presets**: XS, S, M, L, XL
- Approximately 1.5× scaling between sizes
- Larger models improve both final performance and data efficiency

### Main Training Loop

**Entry Point**: `python dreamerv3/main.py`

**Basic Command**:
```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/$(date +%Y%m%d_%H%M%S) \
  --configs crafter \
  --run.train_ratio 32
```

**Training Flow**:

1. **Environment Interaction**:
   - Agent acts in environment using current policy
   - Store transitions `(observation, action, reward, done)` in replay buffer

2. **World Model Update**:
   - Sample batch of sequences from replay buffer
   - Train encoder, dynamics, decoder, reward, continue predictors
   - Minimize reconstruction + dynamics + representation losses

3. **Policy Update**:
   - Sample batch of starting states from replay buffer
   - Generate imagined rollouts using world model
   - Compute λ-returns from imagined trajectories
   - Update actor (maximize returns) and critic (predict returns)

4. **Repeat**: Steps 1-3 continuously

**Train Ratio**: 32 means 32 gradient steps per environment step
- Enables high sample efficiency
- Off-policy learning from replay buffer
- GPU trains while environment runs

### Replay Buffer

- Stores real environment transitions
- Circular buffer with fixed capacity
- Sampled uniformly for training
- Batch sampling: Random episodes, then random starting points

---

## Asynchronous Training (DayDreamer)

### Two-Process Architecture

DayDreamer decouples data collection from learning for real-world robots:

**1. Actor Process** (Robot Control):
- Computes actions in real-time for robot
- Executes actions in environment
- Sends trajectories (128 time steps) to replay buffer
- Receives policy weight updates from learner every 20 seconds
- Can run on CPU (for arms) or GPU (for quadruped)

**2. Learner Process** (GPU Training):
- Continuously samples from replay buffer
- Updates world model on real data
- Optimizes policy via imagination rollouts
- Pushes policy weights to actor periodically
- Runs on GPU for fast training

### Running DayDreamer

**Terminal 1 - Learner**:
```bash
CUDA_VISIBLE_DEVICES=0 python dreamerv2/train.py \
  --logdir ~/logdir/daydreamer/$(date +%Y%m%d_%H%M%S) \
  --configs ur5 \
  --run learning
```

**Terminal 2 - Actor**:
```bash
python dreamerv2/train.py \
  --logdir ~/logdir/daydreamer/$(date +%Y%m%d_%H%M%S) \
  --configs ur5 \
  --run acting \
  --env.kbreset True
```

### Robot-Specific Configurations

**Unitree A1 Quadruped**:
- Control rate: 20 Hz (fast)
- Actions: Continuous (motor angles)
- Inputs: Proprioception (low-dimensional)
- Both learner and actor use GPU
- Learned to walk in 1 hour

**UR5 Robotic Arm**:
- Control rate: ~0.5 Hz
- Actions: Discrete
- Inputs: RGB-D camera + proprioception
- Learner on GPU, actor on CPU (JIT disabled)
- Visual pick-and-place task
- Average pick rate: 2.5 objects/min after 8 hours

**XArm 7-DOF Arm**:
- Control rate: ~0.5 Hz
- Actions: Discrete
- Inputs: RGB-D RealSense camera + proprioception
- Learner on GPU, actor on CPU
- Sensor fusion in world model
- Average pick rate: 3.1 objects/min after 10 hours

### Synchronization

- Policy weights synced every 20 seconds
- Replay buffer shared between processes
- Actor adds trajectories, learner samples
- No blocking: actor continues with stale policy until update

---

## Robustness Techniques

DreamerV3's key innovation is robustness techniques enabling fixed hyperparameters across domains.

### 1. Symlog/Symexp Transformations

**Problem**: Rewards and observations vary in scale across domains (e.g., 0-1 vs 1000+)

**Solution**: Symmetric logarithm function

```
symlog(x) = sign(x) * ln(|x| + 1)
symexp(x) = sign(x) * (exp(|x|) - 1)
```

**Properties**:
- Symmetric around origin (preserves sign)
- Approximates identity near 0
- Compresses large values logarithmically
- No truncation of extreme values
- No non-stationarity from running statistics

**Applications**:
- Transform vector observations (encoder input and decoder target)
- Symexp two-hot loss for reward predictor
- Symexp two-hot loss for critic (return prediction)

### 2. Two-Hot Encoding

For continuous targets (rewards, returns):
- Discretize into 255 bins with exponential spacing
- Soft targets: distribute probability over two adjacent bins
- More stable than direct regression
- Handles multi-modal distributions (critic)

### 3. KL Balance with Free Bits

**Problem**: Encoder-decoder can ignore stochastic states, using only deterministic state

**Solution**: Balanced KL divergence loss
- Mix of forward KL (dynamics → encoder) and backward KL (encoder → dynamics)
- Free bits: minimum KL per categorical to prevent posterior collapse
- Ensures latent states carry information

### 4. 1% Unimix for Categoricals

- Add 1% uniform distribution to categorical outputs
- Prevents overconfident predictions
- Improves exploration and robustness

### 5. Percentile Return Normalization

- Normalize returns by percentile statistics (5th-95th)
- More robust than mean/std normalization
- Adaptive to changing return distribution

### 6. Block GRU

- More efficient than standard GRU
- Better gradient flow
- Scales to larger hidden sizes

### 7. RMSNorm

- Root Mean Square normalization
- Simpler and faster than LayerNorm
- Good stability properties

---

## Configuration Parameters Summary

### Latent Dimensions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `deter` | 8192 | Deterministic (recurrent) state size |
| `stoch` | 32 | Number of categorical distributions |
| `classes` | 64 | Classes per categorical |
| `hidden` | 1024 | MLP hidden layer size |

**Total stochastic state**: 32 × 64 = 2048 possible discrete values (as one-hots)
**Feature dimension**: 8192 + (32×64) = 10,240 for full state

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Parallel sequences |
| `batch_length` | 64 | Time steps per sequence |
| `imag_horizon` | 15 | Imagination rollout length |
| `train_ratio` | 32.0 | Gradient steps per env step |
| `learning_rate` | 4e-5 | Adam learning rate |
| `horizon` | 333 | Episode horizon for λ-returns |

### Network Architecture

| Component | Layers | Units | Activation |
|-----------|--------|-------|------------|
| Encoder CNN | 4 | 64×[2,3,4,4] | SiLu |
| Decoder CNN | 4 | 64×[2,3,4,4] | SiLu |
| Dynamics (RSSM) | Block GRU | 8192 | - |
| Actor MLP | 3 | 1024 | SiLu |
| Critic MLP | 3 | 1024 | SiLu |
| Reward Head | 1 | 1024 | SiLu |

### Model Size Scaling

Presets: `XS`, `S`, `M`, `L`, `XL`

Scaling factors (approximately 1.5× between sizes):
- All dimensions scale together
- Larger = better performance + data efficiency
- XL recommended for complex domains

### Loss Weights

| Loss | Weight | Description |
|------|--------|-------------|
| Decoder (reconstruction) | 1.0 | Image/observation reconstruction |
| Reward | 1.0 | Reward prediction |
| Continue | 1.0 | Episode termination |
| Dynamics | 1.0 | Dynamics predictor accuracy |
| Representation | 0.1 | Encoder regularization |
| Actor | 1.0 | Policy gradient |
| Critic | 1.0 | Value prediction |

---

## Dependencies

### DreamerV3 (JAX)

**Core Requirements**:
```
python >= 3.11
jax >= 0.4.6
jaxlib >= 0.4.6 (with CUDA support)
```

**Installation**:
```bash
# JAX with CUDA 11.8
pip install "jax[cuda11_cudnn82]==0.4.6" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Other dependencies
pip install -r requirements.txt

# Environment-specific (e.g., Atari)
sh dreamerv3/embodied/scripts/install_atari.sh
```

**Key Packages**:
- `numpy` - Numerical computing
- `optax` - JAX optimization library
- `flax` - Neural network library for JAX
- `gymnasium` or `gym` - RL environments
- `imageio` - Video logging
- Environment packages as needed

### DayDreamer (TensorFlow 2)

**Core Requirements**:
```
python >= 3.8
tensorflow >= 2.6.0
tensorflow-probability
```

**Installation**:
```bash
pip install tensorflow tensorflow-probability
pip install -r requirements.txt
```

**Robot-Specific**:
- ROS packages for robot control
- Camera drivers (RealSense SDK for RGB-D)
- Robot-specific control libraries (e.g., PyBullet for simulation)

### Hardware Requirements

**Minimum**:
- GPU: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- RAM: 16GB+ system memory
- CPU: Multi-core for parallel environment simulation

**Recommended**:
- GPU: NVIDIA RTX 3090 or better (24GB VRAM)
- RAM: 32GB+
- CPU: 16+ cores for DayDreamer async training

**For Real Robots**:
- Separate GPU for learner process (training)
- CPU or second GPU for actor process (control)
- Low-latency connection to robot

---

## Integration Approach for Our Project

### Phase 1: Foundation Setup

**1. Environment Setup**
- Install DreamerV3 dependencies (JAX or PyTorch version)
- Set up robot simulation environment (e.g., PyBullet, Isaac Gym)
- Configure camera inputs and proprioception sensors
- Test basic environment interaction loop

**2. World Model Pre-training**
- Collect initial dataset via random exploration or teleoperation
- Train world model on demonstration data
- Validate reconstruction quality and reward prediction
- Establish baseline for imagination accuracy

### Phase 2: VLA Integration

**1. Policy Initialization**
- Load pre-trained VLA weights (π₀.₅ or similar)
- Replace DreamerV3's standard actor with VLA policy network
- Keep vision encoder frozen or lightly fine-tune
- Integrate language instruction conditioning

**2. Language-Conditioned World Model**
- Extend RSSM to accept language embeddings as input
- Concatenate instruction embeddings to state features
- Train on multi-task dataset with diverse instructions
- Test task-specific imagination quality

**3. Hybrid Architecture**
```
Input: (image, proprioception, instruction)
   ↓
VLA Vision Encoder (frozen/light fine-tune)
   ↓
RSSM Dynamics Model (trainable)
   ↓
Combined Features [h_t, z_t, instruction_emb]
   ↓
VLA Policy Head (fine-tune) + Dreamer Critic
```

### Phase 3: RL Fine-tuning

**1. Reward Function Setup**
- Train reward classifier on success/failure demonstrations
- Integrate with Dreamer's reward predictor
- Combine learned rewards with task-specific metrics
- Validate reward signal quality

**2. Hybrid Loss Function**
```python
L_total = L_RL + λ_IL * L_IL + L_world_model

where:
  L_RL = actor_loss + critic_loss (from imagination)
  L_IL = behavior_cloning_loss (on demonstrations)
  L_world_model = reconstruction + dynamics + reward losses
```

**3. Curriculum Training**
- Start with high λ_IL (imitation-heavy)
- Gradually decay λ_IL to shift toward RL
- Increase imagination horizon as world model improves
- Monitor for catastrophic forgetting of VLA capabilities

### Phase 4: Human-in-the-Loop

**1. Intervention System**
- Implement intervention detection and logging
- Store interventions in replay buffer with high priority
- Use interventions for reward model refinement
- Track intervention rate as success metric

**2. Safety Protocols**
- E-stop integration for immediate shutdown
- Workspace boundaries and collision detection
- Conservative policy during early training
- Gradual increase in action magnitude limits

### Phase 5: Asynchronous Real-World Training

**1. DayDreamer-Style Architecture**
- Implement separate actor and learner processes
- Actor: Real-time robot control with current policy
- Learner: Continuous GPU training on replay buffer
- Policy synchronization every 20 seconds

**2. Efficient Training**
- High train_ratio (32-64) for sample efficiency
- Prioritized replay for important transitions
- Batch updates during robot reset times
- Parallel imagination on GPU while robot acts

**3. Real/Imagined Data Balance**
- Train world model exclusively on real data
- Train policy primarily on imagined data
- Mix in real data for policy updates (10-20%)
- Monitor imagination accuracy vs reality

### Phase 6: Multi-Task Scaling

**1. Task Distribution**
- Train on multiple manipulation tasks concurrently
- Language instructions distinguish tasks
- Shared world model and vision encoder
- Task-specific reward heads if needed

**2. Zero-Shot Generalization**
- Test on novel instruction paraphrases
- Evaluate on unseen object configurations
- Measure transfer to related tasks
- Assess compositional understanding

### Key Integration Considerations

**Architecture Decisions**:
- Use PyTorch implementation for easier VLA integration
- Keep world model compact for fast imagination (reduce latent dims if needed)
- Separate reward heads for different task categories
- Language-conditioned value network for multi-task credit assignment

**Training Strategy**:
- Pre-train world model before RL to ensure quality imagination
- Use small imagination horizon initially (5-10 steps)
- Increase horizon gradually as model accuracy improves
- High train_ratio essential for real-robot sample efficiency

**Data Management**:
- Large replay buffer (100K+ transitions)
- Prioritize diverse state coverage
- Oversample rare success cases
- Save intervention trajectories permanently

**Hyperparameter Priorities**:
1. `train_ratio`: 32-64 for sample efficiency
2. `imag_horizon`: Start 5-10, increase to 15-20
3. `λ_IL` decay schedule: 1.0 → 0.1 over training
4. `batch_size`: 16-32 for stable gradients
5. Model size: Start with S or M, scale up if needed

### Expected Performance Gains

Based on DayDreamer paper and DreamerV3 results:

- **Sample Efficiency**: 5-10× fewer real-world interactions vs pure RL
- **Training Time**: Hours instead of days for basic manipulation
- **Success Rate**: 80-95% on practiced tasks with good reward signal
- **Generalization**: Strong zero-shot to instruction variations
- **Safety**: Minimal interventions (<5%) after initial training phase

### Repository Structure for Integration

```
/models
  /vla
    - vision_encoder.py (frozen/light fine-tune)
    - language_encoder.py (frozen)
    - policy_head.py (fine-tune)
  /world_model
    - rssm.py (RSSM with language conditioning)
    - encoder.py (observation encoder)
    - decoder.py (observation decoder)
    - reward.py (reward predictor)
    - continue.py (termination predictor)
  /reward
    - classifier.py (success/failure discrimination)

/training
  /imitation
    - pretrain_vla.py (Stage 1)
  /rl
    - dreamer_trainer.py (world model + actor-critic)
    - replay_buffer.py (experience storage)
    - hybrid_loss.py (RL + IL combined)
  /world_model
    - wm_trainer.py (separate world model training)

/robot
  /control
    - actor_process.py (real-time control)
    - learner_process.py (async training)
    - intervention.py (human takeover)
  /sensors
    - camera.py (RGB-D capture)
    - proprioception.py (joint states)

/evaluation
  /metrics
    - success_rate.py
    - intervention_rate.py
  /baselines
    - il_only.py
    - rl_no_world_model.py

/configs
  - dreamer_config.yaml (DreamerV3 hyperparameters)
  - vla_config.yaml (VLA architecture and weights)
  - robot_config.yaml (hardware specifications)
  - task_config.yaml (task definitions and instructions)
```

---

## Key Papers and Resources

### Primary References

1. **DreamerV3**: "Mastering Diverse Domains through World Models"
   - Hafner et al., 2023
   - https://arxiv.org/abs/2301.04104
   - Published in Nature (2025)

2. **DayDreamer**: "World Models for Physical Robot Learning"
   - Wu, Escontrela, Hafner, et al., 2022
   - https://arxiv.org/abs/2206.14176
   - CoRL 2022

3. **DreamerV2**: "Mastering Atari with Discrete World Models"
   - Hafner et al., 2021
   - ICLR 2021

4. **Dreamer**: "Dream to Control: Learning Behaviors by Latent Imagination"
   - Hafner et al., 2020
   - ICLR 2020

### Useful Implementations

- Official JAX: https://github.com/danijar/dreamerv3
- Official TF2 (DayDreamer): https://github.com/danijar/daydreamer
- PyTorch (clean): https://github.com/NM512/dreamerv3-torch
- PyTorch (full): https://github.com/burchim/DreamerV3-PyTorch

### Additional Reading

- DreamerV3 project page: https://danijar.com/project/dreamerv3/
- DayDreamer project page: https://danijar.com/project/daydreamer/
- RLlib implementation: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html

---

## Next Steps for Our Project

### Immediate Actions

1. **Install and Test DreamerV3**
   - Set up PyTorch implementation (NM512/dreamerv3-torch)
   - Test on simple simulation environment (e.g., DeepMind Control Suite)
   - Verify GPU utilization and training speed
   - Benchmark imagination vs reality accuracy

2. **Explore VLA Integration Points**
   - Identify VLA model architecture (π₀.₅ or alternative)
   - Map VLA components to DreamerV3 structure
   - Design language conditioning mechanism
   - Plan encoder freezing strategy

3. **Design Reward Function**
   - Collect initial demonstration dataset
   - Train binary success classifier
   - Test reward signal quality
   - Plan human intervention protocol

4. **Prototype Hybrid System**
   - Implement combined loss (RL + IL)
   - Test on simple manipulation task
   - Validate learning signal strength
   - Measure sample efficiency gains

### Validation Experiments

1. **World Model Quality**
   - Image reconstruction accuracy
   - Reward prediction correlation with ground truth
   - Multi-step prediction error over horizon
   - Language-conditioned prediction accuracy

2. **Policy Performance**
   - Success rate on trained tasks
   - Zero-shot generalization to new instructions
   - Robustness to visual perturbations
   - Sample efficiency vs baselines (IL-only, RL-no-model)

3. **Real-World Feasibility**
   - Control frequency requirements
   - Latency measurements (actor process)
   - Training throughput (learner process)
   - Intervention frequency over time

### Success Criteria

- World model: <10% prediction error over 10-step horizon
- Policy: >80% success rate on practiced tasks
- Sample efficiency: 5× fewer real interactions than pure RL
- Generalization: >60% success on paraphrased instructions
- Safety: <5% intervention rate after 10 hours of training
- Training speed: Minutes per gradient step (including imagination)

---

## Conclusion

DreamerV3 and DayDreamer provide a solid foundation for integrating world models with our VLA-based manipulation system. The key advantages are:

1. **Sample Efficiency**: Imagination rollouts dramatically reduce real-world data requirements
2. **Robustness**: Fixed hyperparameters work across domains via normalization techniques
3. **Scalability**: Proven on real robots with different control frequencies and sensors
4. **Modularity**: Clear separation of world model, actor, and critic components
5. **Performance**: State-of-the-art results on diverse benchmarks

The asynchronous actor-learner architecture from DayDreamer is particularly well-suited for real robot training, enabling continuous learning without blocking robot execution.

Our integration approach leverages these strengths while preserving the benefits of VLA pre-training (language understanding, broad task coverage) and RL fine-tuning (task-specific optimization). The hybrid loss function ensures we don't lose VLA capabilities while gaining sample efficiency from world model imagination.

The next phase involves hands-on implementation, starting with the world model on simulated tasks, then gradually integrating VLA components and moving toward real robot training with human-in-the-loop safety.
