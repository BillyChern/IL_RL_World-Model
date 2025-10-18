# IL + RL + World Model for Robotic Manipulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research project integrating **Vision-Language-Action (VLA) models**, **Human-in-the-Loop Reinforcement Learning**, and **World Models** for sample-efficient robot manipulation learning.

## Overview

This system combines three cutting-edge approaches:
1. **π0.5 VLA** - Pre-trained vision-language-action model for broad generalization
2. **HIL-SERL** - Human-in-the-loop RL for precise, safe skill acquisition
3. **DreamerV3** - World model for parallel imagination and accelerated training

**Key Innovation**: While the robot performs 1 real trial, the world model simulates 10+ imagined trials, dramatically reducing real-world data requirements while maintaining human-level performance.

## Key Features

- **Bi-manual manipulation** with dual ARX X5 robot arms
- **Language-conditioned policies** for multi-task learning
- **Real-time RL training** with asynchronous robot collection and GPU training
- **Human interventions** for safety and corrective demonstrations
- **8x H100 GPU** distributed training support
- **Comprehensive logging** with Weights & Biases

## Installation

### Prerequisites
- Python 3.11+
- CUDA 12.0+
- 8x H100 GPUs (80GB each)
- ARX X5 robot arms with SDK

### Setup

```bash
# Clone repository
git clone https://github.com/BillyChern/IL_RL_World-Model.git
cd IL_RL_WorldModel

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up pre-trained weights (downloads π0.5 checkpoints)
python scripts/download_weights.py
```

## Quick Start

### 1. Prepare Demonstration Data

```bash
# Convert your demos to LeRobot RLDS format
python scripts/prepare_data.py \
    --input_dir /path/to/raw/demos \
    --output_dir data/demonstrations \
    --tasks "pick_and_place,fold_towel"
```

### 2. Pre-train VLA Policy

```bash
# Fine-tune π0.5 on your demonstrations
python scripts/train_vla.py \
    --config configs/vla_training.yaml \
    --data_dir data/demonstrations
```

### 3. Train Reward Classifier

```bash
# Train binary reward classifier from demo success/failure labels
python scripts/train_reward.py \
    --config configs/reward_classifier.yaml \
    --demo_dir data/demonstrations
```

### 4. Run Integrated Training

```bash
# Launch asynchronous training: robot collection + RL + world model
python scripts/train_integrated.py \
    --config configs/integrated_training.yaml \
    --vla_checkpoint checkpoints/vla_pretrained.pth \
    --reward_checkpoint checkpoints/reward_classifier.pth
```

### 5. Evaluate Policy

```bash
# Test on held-out tasks
python scripts/evaluate.py \
    --checkpoint checkpoints/integrated_final.pth \
    --num_episodes 50 \
    --tasks "pick_and_place,fold_towel"
```

## Project Structure

```
IL_RL_WorldModel/
├── models/
│   ├── vla/                 # π0.5 VLA integration
│   ├── world_model/         # DreamerV3 world model
│   └── reward/              # Reward classifier
├── training/
│   ├── imitation/           # Behavioral cloning
│   ├── rl/                  # HIL-SERL RL pipeline
│   └── world_model/         # World model training
├── robot/
│   ├── arx_x5/              # ARX X5 interface (50Hz control)
│   └── teleop/              # Human intervention system
├── data/
│   ├── demonstrations/      # RLDS demonstration data
│   ├── replay_buffer/       # RL experience buffer
│   └── imagination/         # World model rollouts
├── evaluation/
│   └── metrics/             # Success rate, timing metrics
├── configs/                 # Hydra configuration files
├── scripts/                 # Training and evaluation scripts
├── tests/                   # Unit tests
└── docs/                    # Codebase exploration docs
```

## Configuration

All training parameters are specified in YAML configs under `configs/`. Key parameters:

```yaml
# Imagination settings
world_model:
  imagination_horizon: 10      # Start at 5-10, increase as model improves
  real_imagined_ratio: 0.8     # 80% real, 20% imagined initially

# RL settings
rl:
  algorithm: sac               # Soft Actor-Critic
  hybrid_loss_weight: 0.1      # λ for L = L_RL + λ*L_IL (decays)
  intervention_rate: 0.3       # Target ~30% initially, reduce to ~0%

# VLA settings
vla:
  freeze_vision: true          # Freeze pre-trained vision encoder
  freeze_language: true        # Freeze pre-trained language encoder
  action_dim: 14               # 7 per arm (6 joints + gripper)
```

## Hardware Specifications

- **Robot**: Dual ARX X5 arms (6 DoF + gripper each)
- **Cameras**: 3 fish-eye cameras (left, right, base)
- **Control**: 50Hz joint position control via ARX X5 SDK
- **Compute**: 8x H100 GPUs (80GB VRAM each)

## Training Pipeline

1. **Stage 1**: Imitation Learning pre-training on demonstrations
2. **Stage 2**: Reward classifier training
3. **Stage 3**: World model pre-training on demo data
4. **Stage 4**: Integrated RL training with human interventions
5. **Stage 5**: Evaluation and multi-task scaling

## Citations

This project builds on multiple state-of-the-art works. Please see [CITATIONS.md](CITATIONS.md) for full references.

**Core papers**:
- π0.5 VLA (Physical Intelligence, 2025)
- HIL-SERL (Luo et al., CoRL 2024)
- DreamerV3 (Hafner et al., 2023)
- DayDreamer (Wu et al., CoRL 2023)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This is a research project with publication goals. Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## Acknowledgments

We gratefully acknowledge:
- Physical Intelligence for open-sourcing π0.5
- UC Berkeley RAIL Lab for HIL-SERL
- Danijar Hafner for DreamerV3/DayDreamer
- Hugging Face for LeRobot
- ARX Robotics for the X5 SDK

## Contact

- **Author**: Billy Chern (Shichen)
- **Email**: shichen22@m.fudan.edu.cn
- **GitHub**: [BillyChern](https://github.com/BillyChern)

## Project Status

🚧 **In Active Development** 🚧

Current progress:
- [x] Codebase exploration (OpenPI, DreamerV3, HIL-SERL, LeRobot, ARX X5)
- [x] Repository structure and documentation
- [x] **ARX X5 robot interface** - DualARXController with 50Hz control, safety limits, 3-camera system (17 tests ✓)
- [x] **Data pipeline** - RLDS dataset loader, replay buffer with demo/online mixing (15 tests ✓)
- [x] **π0.5 VLA integration** - Vision-language-action model with diffusion decoder (4 tests ✓)
- [x] **Reward classifier** - Binary success/failure predictor with ResNet-10
- [x] **DreamerV3 world model** - RSSM implementation in JAX for imagination rollouts
- [x] **SAC RL algorithm** - Actor-critic networks with HIL-SERL hybrid loss
- [x] **Human intervention system** - Keyboard-based control for safety and corrections
- [x] **Asynchronous training** - Dual-loop: robot collection + GPU training threads
- [x] **Weights & Biases integration** - Full logging with API key configured
- [x] **Multi-GPU distributed training** - 8x H100 support with PyTorch DDP + JAX pmap
- [x] **Main training script** - Integrated pipeline with all components
- [ ] Evaluation scripts and benchmarking
- [ ] Full integration testing with hardware

**Tests**: 36 passing | **Code**: ~8000+ lines | **Docs**: 161KB exploration + architecture

See [CLAUDE.md](CLAUDE.md) for detailed implementation notes and [docs/](docs/) for exploration documentation.
