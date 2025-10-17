# Citations and Acknowledgments

This project integrates multiple state-of-the-art research works and open-source projects. We gratefully acknowledge the following contributions:

## Core Research Papers

### Vision-Language-Action Models
```bibtex
@article{black2024pi0,
  title={π₀: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, David and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024},
  url={https://www.physicalintelligence.company/download/pi0.pdf}
}

@misc{physical_intelligence_2025_pi05,
  title={π₀.₅: Improved Open-World Generalization for Embodied AI},
  author={Physical Intelligence},
  year={2025},
  url={https://www.physicalintelligence.company/blog/pi05}
}
```

### Human-in-the-Loop Reinforcement Learning
```bibtex
@inproceedings{luo2024hilserl,
  title={HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Luo, Jianlan and Xu, Zheyuan and Haas, Charles and Levine, Sergey and Zheng, Quan},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024},
  url={https://hil-serl.github.io/}
}
```

### World Models
```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023},
  url={https://danijar.com/project/dreamerv3/}
}

@inproceedings{wu2023daydreamer,
  title={DayDreamer: World Models for Physical Robot Learning},
  author={Wu, Philipp and Escontrela, Alejandro and Hafner, Danijar and Abbeel, Pieter and Goldberg, Ken},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2023},
  url={https://danijar.com/project/daydreamer/}
}
```

## Open-Source Code and Tools

### OpenPI (π₀ and π₀.₅)
- **Repository**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- **License**: MIT License
- **Usage**: VLA policy architecture, pre-trained weights, training pipeline
- **Citation**: See π₀ and π₀.₅ papers above

### DreamerV3
- **Repository**: [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- **License**: MIT License
- **Author**: Danijar Hafner
- **Usage**: World model architecture, RSSM implementation, training algorithms
- **Citation**: See Hafner et al. 2023 above

### HIL-SERL
- **Repository**: [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl)
- **License**: MIT License
- **Authors**: Jianlan Luo, Charles Haas, and UC Berkeley RAIL Lab
- **Usage**: RL training pipeline, human intervention system, replay buffer design
- **Citation**: See Luo et al. 2024 above

### LeRobot
- **Repository**: [huggingface/lerobot](https://github.com/huggingface/lerobot)
- **License**: Apache 2.0
- **Maintainer**: Hugging Face
- **Usage**: RLDS dataset format, data loading utilities, robot datasets
- **Citation**:
```bibtex
@software{cadene2024lerobot,
  title={LeRobot: State-of-the-art Machine Learning for Real-World Robotics},
  author={Cadene, Remi and others},
  year={2024},
  publisher={Hugging Face},
  url={https://github.com/huggingface/lerobot}
}
```

### ARX X5 SDK
- **Repository**: [ARXroboticsX/ARX_X5](https://github.com/ARXroboticsX/ARX_X5)
- **License**: BSD-3-Clause
- **Manufacturer**: ARX Robotics
- **Usage**: Robot control interface, proprioception, safety features

## Additional Libraries and Frameworks

### Core ML Frameworks
- **JAX**: [google/jax](https://github.com/google/jax) - Apache 2.0
- **PyTorch**: [pytorch/pytorch](https://github.com/pytorch/pytorch) - BSD-3-Clause
- **Transformers**: [huggingface/transformers](https://github.com/huggingface/transformers) - Apache 2.0

### Vision and Language Models
- **SigLIP**: Google Research - Used for vision encoding in π₀.₅
- **Gemma**: Google DeepMind - Language model backbone
- **CLIP**: OpenAI - Vision-language pre-training

### Robotics and RL
- **Gymnasium**: [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - MIT
- **Stable-Baselines3**: [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) - MIT

### Monitoring and Logging
- **Weights & Biases**: [wandb/wandb](https://github.com/wandb/wandb) - MIT

## Datasets

### DROID (Distributed Robot Interaction Dataset)
```bibtex
@article{khazatsky2024droid,
  title={DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset},
  author={Khazatsky, Alexander and Pertsch, Karl and Nair, Suraj and Balakrishna, Ashwin and Dasari, Sudeep and Karamcheti, Siddharth and Nasiriany, Soroush and Srinivasan, Mohan and Zhu, Yuke and Finn, Chelsea and Levine, Sergey and Sadigh, Dorsa},
  journal={arXiv preprint arXiv:2403.12945},
  year={2024}
}
```

## Related Influence and Inspiration

- **RT-X**: Robotic Transformer X-Embodiment - Cross-embodiment robot learning
- **Octo**: Open-source generalist robot policies
- **V-JEPA**: Joint-embedding predictive architecture for video understanding

---

## How to Cite This Project

If you use this codebase in your research, please cite:

```bibtex
@software{chern2025_il_rl_world_model,
  title={Integrating Imitation Learning, Real-Time RL, and World Models for Robotic Manipulation},
  author={Chern, Billy (Shichen)},
  year={2025},
  url={https://github.com/BillyChern/IL_RL_World-Model},
  note={Integration of π₀.₅ VLA, HIL-SERL RL, and DreamerV3 world models}
}
```

And please also cite the original papers for π₀.₅, HIL-SERL, and DreamerV3 listed above.

---

*Last updated: October 2025*

*This file is maintained to properly credit all contributors to the open-source robotics community. If you notice any missing attributions, please open an issue.*
