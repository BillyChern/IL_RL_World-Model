# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project integrating three cutting-edge approaches for robotic manipulation:

1. **Vision-Language-Action (VLA) agent** - Trained via imitation learning for broad task generalization and language understanding (based on π₀.₅/pi0.5)
2. **Real-time Reinforcement Learning (RL)** - To boost success rates and adaptability (HIL-SERL approach)
3. **Learned World Models** - To simulate experiences in parallel and accelerate training (DayDreamer/Dreamer approach)

The goal is to dramatically reduce dependence on human-provided demonstrations while achieving robust, real-world performance on complex manipulation tasks.

## Development Workflow

**CRITICAL REQUIREMENTS for all development:**

### 1. Test-Driven Development
- **ALWAYS write unit tests** for each component BEFORE considering it complete
- Minimum test coverage: Every public function/method must have at least one test
- Tests must be comprehensive and actually verify correctness
- Run tests after every implementation: `pytest tests/ -v`
- Target: 80%+ code coverage for all modules

### 2. Git Commit Discipline
- **Commit FREQUENTLY** - after completing each logical unit of work
- Commit pattern:
  - After implementing a new module/class
  - After writing tests that pass
  - After updating documentation
  - After fixing bugs
  - **At least every 30-60 minutes of work**
- Commit message format: `<type>: <description>`
  - Types: feat, fix, test, docs, refactor, chore
  - Examples:
    - `feat: Add DreamerV3 RSSM world model`
    - `test: Add comprehensive tests for replay buffer`
    - `docs: Update README with installation instructions`

### 3. Documentation Updates
- **Update README.md** whenever:
  - New features are added
  - Installation steps change
  - Usage examples need updating
- **Update ARCHITECTURE.md** when system design changes
- **Keep CLAUDE.md current** with development practices
- All docstrings must follow Google style
- Add inline comments for complex logic

### 4. Verification Before Proceeding
- Before moving to the next component:
  1. ✓ All tests passing
  2. ✓ Code committed to git
  3. ✓ Documentation updated
  4. ✓ No redundant code
  5. ✓ Clean, readable implementation

### 5. Branch Strategy
- **main branch**: Stable, tested code only
- **dev branch**: Active development (use this!)
- Feature branches: For major new features
- Always work on dev or feature branches
- Only merge to main after thorough testing

### 6. Code Review Checklist
Before committing, verify:
- [ ] Code follows project style (clean, no redundancy)
- [ ] All new functions have docstrings
- [ ] Unit tests written and passing
- [ ] No hardcoded paths or credentials
- [ ] Imports organized and minimal
- [ ] Type hints where applicable
- [ ] Error handling implemented

## Project Architecture

The system follows a multi-component architecture:

### Core Components

**VLA Policy Module**
- Vision-language-action model accepting visual observations + language instructions
- Initialized with pre-trained weights (vision transformer + language model)
- Outputs high-level action commands or low-level controls
- Must support real-time inference with GPU acceleration

**RL Training Pipeline**
- Off-policy actor-critic method (Dreamer variant or SAC with experience replay)
- Replay buffer containing: state observations (camera images, proprioception), instructions, actions, rewards
- Hybrid loss function: L = L_RL + λ * L_IL (combines RL loss with imitation loss)
- Language-conditioned policy: π(a | s, instruction)
- Human-in-the-loop interventions during training for safety

**World Model**
- Two potential architectures:
  - Latent dynamics model (DreamerV3-style): CNN encoder + GRU/transformer dynamics + MLP reward decoder
  - Video generation model (Diffusion or JEPA): predicts next image frames given current frame + action
- Accepts: current state (image + proprioception) + action + optional language instruction
- Outputs: prediction of next state (or latent encoding) + reward
- Enables faster-than-real-time parallel simulation in "dream" environment

**Reward Function**
- Binary reward classifier trained on teleoperated demonstrations
- Distinguishes successful outcomes vs. failures from observations
- May combine learned signal with task-specific metrics (e.g., distance to target)

### Data Flow

1. **Real Robot Loop** (asynchronous thread):
   - Reset environment → Current policy generates actions → Execute → Log to replay buffer

2. **Training Loop** (asynchronous thread):
   - World Model Update: Train on latest replay buffer data
   - Imaginary Rollouts: Generate simulated trajectories using world model
   - Policy Update: Sample from mix of real + imagined transitions

3. **Parallel Training**:
   - While robot performs 1 real trial, world model simulates 10+ imagined trials
   - Imagined transitions stored in separate imagination buffer
   - Policy trained on mixture of real and imagined data

## Development Workflow

### Training Stages

**Stage 1: Imitation Learning Pre-training**
- Train/fine-tune VLA on heterogeneous data (image-text pairs, teleoperation trajectories, instruction-demo pairs)
- Initialize from pre-trained vision-language models
- Validate generalization on unseen environments before RL

**Stage 2: RL Setup**
- Initialize policy network with pre-trained VLA weights
- Freeze or partially freeze vision/language encoders
- Set up reward classifier from demonstration data
- Configure human intervention protocol

**Stage 3: World Model Training**
- Pre-train on demonstration set + exploratory data
- If using V-JEPA 2 or similar, fine-tune on robot-specific data
- Continuously update online as new episodes arrive

**Stage 4: Integrated Training**
- Run asynchronous dual-loop: real robot collection + GPU training
- Start with short imagination horizon (5-10 steps), extend as model improves
- Begin with conservative mixing (80% real, 20% imagined), shift to 50-50 or more
- Gradually reduce human intervention frequency as policy improves

**Stage 5: Evaluation & Scaling**
- Quantitative: success rate, task completion time vs. baselines
- Qualitative: zero-shot new instructions, perturbation resilience
- Test generalization to new tasks with minimal/no demonstrations

### Key Training Parameters

- **Imagination horizon (H)**: Start at 5-10 steps, increase as world model accuracy improves
- **Real/imagined data mixing ratio**: Start 80/20, shift toward 50/50 or more imagined
- **IL loss weight (λ)**: Decay over time to transition from imitation-guided to RL-driven
- **Update frequency**: High gradient steps per real episode (off-policy RL characteristic)
- **Intervention rate**: Start ~30%, target near 0% by end of training

## Technical Considerations

### Multi-Task and Language Conditioning

- Policy and value networks must handle instruction embeddings concatenated to state
- Train on multiple tasks concurrently, distinguished by language instruction
- Reward alignment: success only counted for completing the commanded task
- Augment with paraphrased instructions to avoid overfitting to specific phrasing

### World Model Fidelity

- Continuously update on latest data to track environment dynamics
- Monitor for model bias/exploitation (agent finding world model errors)
- Use model uncertainty estimation (ensembles/variance) to down-weight uncertain predictions
- Optionally condition on task instruction to focus predictions on goal-relevant aspects

### Safety and Human Interventions

- Human operator intervenes when robot approaches harmful/incorrect actions
- Interventions added to replay buffer as corrective demonstrations
- Protocol: frequent early interventions → gradual reduction as policy improves
- Evaluation checkpoints with no human help to gauge true autonomous performance

## References

Key papers and systems informing this architecture:

- **π₀.₅ (pi0.5)**: Vision-Language-Action model with open-world generalization
- **HIL-SERL**: Human-in-the-loop RL for precise robotic manipulation
- **DayDreamer**: Real-world robot training with online RL and world models
- **Dreamer/DreamerV3**: Latent dynamics models for efficient RL
- **UVA**: Unified Video Action model integrating video generation with policy learning
- **V-JEPA 2**: Joint-embedding predictive architecture for fast video prediction (30× speedup)

## Expected Project Structure

As implementation proceeds, expect these modules:

```
/models
  /vla          # Vision-Language-Action policy
  /world_model  # Dreamer or video prediction model
  /reward       # Learned reward classifier

/training
  /imitation    # Behavioral cloning pre-training
  /rl           # RL fine-tuning pipeline
  /world_model  # World model training

/data
  /demonstrations  # Teleoperated trajectories
  /replay_buffer   # RL experience storage
  /imagination     # Imagined trajectories

/evaluation
  /metrics      # Success rate, completion time tracking
  /baselines    # IL-only, RL-without-world-model comparisons

/robot
  /control      # Robot interface (ROS or direct control)
  /sensors      # RGB-D camera, proprioception
```

## Hardware and Environment Specifications

### Robot Platform
- **Robot**: Dual ARX X5 arms (bi-manipulation capable)
- **DoF**: 6 DoF + gripper per arm = 7D action space per arm
- **Control**: Joint position control at 50Hz via ARX X5 SDK
- **Proprioception**: Joint angles, velocities, currents (torque proxy), gripper state

### Camera Setup
- **Three fish-eye cameras**: left, right, base
- **Same specifications as π0.5 pre-training**
- **Resolution**: 224x224 (after processing)
- **Format**: RGB images

### Compute Resources
- **GPUs**: 8x H100 (80GB each)
- **Training**: Real-time updates during episodes
- **Distributed**: Multi-GPU training from start
- **Separate allocation**: Different GPUs/CPUs for robot collection vs training

### Data Format
- **Demonstration format**: RLDS (via LeRobot)
- **Structure**: (observations, actions, language_instructions) tuples
- **Language**: Natural language (e.g., "fold the yellow towel"), NOT pre-defined strings
- **Tasks**: Pick-and-place, towel folding, and other tasks from π0.5/HIL-SERL benchmarks

## Implementation Specifications

### VLA Base Model
- **Model**: π0.5 from Physical Intelligence OpenPI repository
- **Source**: https://github.com/Physical-Intelligence/openpi
- **Pretrained weights**: Use official π0.5 checkpoints
- **Action dimensions**: 7D per arm (6 joints + gripper) for ARX X5

### World Model
- **Architecture**: DreamerV3 (latent dynamics model, RSSM)
- **Starting horizon**: 5-10 steps
- **Training frequency**: Follow DayDreamer setup
- **Latent dimensions**: Default from DayDreamer

### RL Algorithm
- **Framework**: HIL-SERL pipeline
- **Algorithm**: SAC (Soft Actor-Critic) with demonstrations
- **Hybrid loss**: L = L_RL + λ * L_IL (λ decays over time)
- **Replay buffer**: Mix of demos, real episodes, and human interventions

### Human Interventions
- **Mechanism**: Keyboard control (abstract interface for external joint commands)
- **Detection**: External joint command received
- **Storage**: Interventions added to replay buffer as corrective demonstrations
- **Protocol**: Start ~30% intervention rate, reduce to ~0%

### Training Setup
- **Training mode**: Real-time (update during episodes, not between)
- **Data mixing**: Start 80% real / 20% imagined, shift to 50-50
- **Asynchronous**: Robot collection and GPU training on same machine, different resources
- **Logging**: Weights & Biases for monitoring and visualization

### Multi-Task Training
- **Strategy**: All tasks trained simultaneously (no curriculum for now)
- **Distinction**: Tasks distinguished by language instruction
- **Sampling**: Uniform task sampling during training

## Development Standards

### Code Quality
- **Clean code**: NO redundant code, well-structured, easy to read
- **Documentation**: Google-style docstrings
- **Package management**: uv
- **Testing**: Unit tests for all components (write, test, modify until passing)
- **Version control**: Git with main + dev branches

### License and Citations
- **License**: MIT
- **Citations**: CITATIONS.md tracks all borrowed code and papers
- **Proper attribution**: Cite OpenPI, DreamerV3, HIL-SERL, LeRobot, ARX X5 SDK

### Repository Organization
- **Structure**: Clean, modular, well-organized
- **README**: Concise, clear, easy to understand
- **Documentation**: docs/ folder contains exploration findings
- **No redundancy**: Single source of truth for each component

## Notes for Implementation

- Vision and language encoders should be frozen or lightly fine-tuned to preserve pre-trained features
- Use GPU acceleration for both policy inference (real-time requirement) and world model simulation
- Implement asynchronous training to maximize sample efficiency (robot collects while GPU trains)
- Log all interventions, failures, and novel scenarios for continuous improvement
- Consider curriculum learning: start with simpler tasks, gradually add complexity (future work)
- This is a research project with serious publication goals - code must be flawless and well-documented
- All contributions will be open-sourced to the robotics community

## Quick Reference

### Key Commands
```bash
# Install dependencies
uv sync

# Train VLA policy (imitation learning)
python scripts/train_vla.py --config configs/vla_training.yaml

# Train world model
python scripts/train_world_model.py --config configs/world_model.yaml

# Run integrated RL training
python scripts/train_integrated.py --config configs/integrated_training.yaml

# Evaluate policy
python scripts/evaluate.py --checkpoint path/to/checkpoint

# Run unit tests
pytest tests/
```

### Important Files
- `CITATIONS.md`: All papers and code we borrow from
- `docs/`: Exploration findings for OpenPI, DreamerV3, HIL-SERL, LeRobot, ARX X5
- `configs/`: Hydra configuration files
- `models/vla/`: π0.5 integration and adaptation
- `models/world_model/`: DreamerV3 world model
- `robot/arx_x5/`: ARX X5 SDK interface
- `training/rl/`: HIL-SERL RL pipeline
