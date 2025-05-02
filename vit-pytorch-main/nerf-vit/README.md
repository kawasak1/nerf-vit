# NeRF-ViT: Vision Transformer Enhanced Neural Radiance Fields

This project explores the integration of Vision Transformers (ViT) with Neural Radiance Fields (NeRF) to improve single-view novel view synthesis. We implement and compare three different architectures that leverage ViT features in the NeRF framework.

## Architecture Options

We implement and compare the following three architectural variants:

1. **Feature Conditioning:** Feed image features as additional conditioning to the NeRF MLP
2. **Transformer-based NeRF:** Replace parts of the FC layers with transformer blocks for better spatial reasoning
3. **Cross-Attention Mechanism:** Use cross-attention to relate 3D points to image features

## Requirements

The project has the following dependencies:

```
pytorch>=1.7.0
torchvision>=0.8.1
matplotlib
numpy
imageio
configargparse
tqdm
tensorboard
einops
```

## Project Structure

```
nerf-vit/
├── configs/
│   └── config.txt              # Configuration file for training
├── models/
│   ├── vit.py                  # Vision Transformer implementation
│   └── nerf_vit.py             # NeRF models with ViT integration
├── utils/
│   └── ... (utility files)
├── train_nerf_vit.py           # Main training script
├── run_comparison.py           # Script to compare different models
└── README.md                   # This file
```

## Training

To train a specific model, run:

```bash
python train_nerf_vit.py --config configs/config.txt --expname [experiment_name] --model_type [model_type]
```

Where `[model_type]` is one of:
- `base` (Baseline NeRF without ViT)
- `feature_conditioning` (Feature Conditioning NeRF)
- `transformer` (Transformer-based NeRF)
- `cross_attention` (Cross-Attention NeRF)

## Evaluation

To evaluate a trained model on the test set:

```bash
python train_nerf_vit.py --config configs/config.txt --expname [experiment_name] --model_type [model_type] --render_only --render_test
```

## Comparing All Models

To train, evaluate, and compare all models in a single command:

```bash
python run_comparison.py --train --evaluate --compare
```

Or individually:

```bash
# Just train all models
python run_comparison.py --train

# Just evaluate all models
python run_comparison.py --evaluate

# Just compare results (after training and evaluation)
python run_comparison.py --compare
```

The comparison results will be saved in the `comparison_results` directory.

## Implementation Details

### Feature Conditioning NeRF

The Feature Conditioning NeRF extracts global features from the input image using a Vision Transformer and conditions the NeRF MLP with these features. Specifically:

1. A ViT processes the input image and outputs a global feature vector
2. This feature vector is projected to the desired dimension and concatenated with the input coordinates
3. The rest of the NeRF MLP processes this conditioned input to predict density and color

### Transformer-based NeRF

The Transformer-based NeRF replaces part of the MLP layers with transformer blocks that attend to ViT features:

1. A ViT processes the input image and outputs patch-wise features
2. The first half of the NeRF network consists of standard MLP layers
3. The second half is replaced with transformer blocks that use cross-attention to relate point features to image features
4. The final output is processed through the standard view-dependent branch of NeRF

### Cross-Attention NeRF

The Cross-Attention NeRF uses cross-attention to directly relate 3D points to image features:

1. A ViT processes the input image and outputs patch-wise features
2. The 3D point features are first processed with an MLP
3. A cross-attention layer attends from point features to image features
4. The attended features are further processed with another MLP and the standard view-dependent branch of NeRF

## Expected Results

Based on our experiments, we expect to observe the following comparative performance (from best to worst):

1. Cross-Attention NeRF - Best performance due to direct relation between 3D points and image features
2. Transformer-based NeRF - Good performance with better spatial reasoning
3. Feature Conditioning NeRF - Improved over baseline but limited by global conditioning
4. Baseline NeRF - Standard performance without ViT features

The actual results may vary depending on the specific scenes and training configurations.

## Citation

If you find this project useful, please consider citing our work:

```
@misc{nerfvit2023,
  author = {Anonymous},
  title = {NeRF-ViT: Vision Transformer Enhanced Neural Radiance Fields},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/anonymous/nerf-vit}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NeRF implementation is based on the original [NeRF PyTorch implementation](https://github.com/yenchenlin/nerf-pytorch)
- The Vision Transformer implementation is adapted from [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch) 