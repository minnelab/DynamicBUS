## BUV Video Generation Model

This is the Breast Ultrasound Video (BUV) generation model for the DynamicBUS project, based on the Latte (Latent Diffusion Transformer) architecture.

### Structure

```
.
├── configs/buv/           # BUV training and sampling configurations
├── datasets/              # BUV dataset loaders
├── diffusers/             # Diffusion model components
├── diffusion/             # Diffusion utilities
├── eval/                  # Evaluation scripts
├── ldm/                   # Latent diffusion model modules
├── models/                # Model architectures
├── sample/                # Sampling scripts
├── tools/                 # Utility tools
├── train_scripts/         # Training shell scripts
├── train_video.py         # Main training script for video
├── train.py               # Base training script
└── utils.py               # Utility functions
```

### Setup

```bash
conda env create -f environment.yml
conda activate latte
```

### Training

To train the BUV video generation model:

```bash
bash train_scripts/buv_train_video.sh
```

For training without class conditioning:

```bash
bash train_scripts/buv_train_video_no_class.sh
```

### Sampling

To generate BUV videos:

```bash
bash sample/buv_video.sh
```

For image-to-video generation:

```bash
bash sample/buv_video_buv_i2v.sh
```

### Contact

**Yaofang Liu**: [https://github.com/Yaofang-Liu](https://github.com/Yaofang-Liu)

### Acknowledgments

This implementation is based on [Latte: Latent Diffusion Transformer for Video Generation](https://github.com/maxin-cn/Latte).

### License

Copyright 2024 Yaofang Liu. Licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.
