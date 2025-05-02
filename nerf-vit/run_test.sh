#!/bin/bash

python train_nerf_vit.py --config configs/config.txt --expname lego_base --model_type base --render_only --render_test --ft_path ./logs/lego_base/200000.tar --testskip 1
python train_nerf_vit.py --config configs/config.txt --expname lego_feature_conditioning --model_type feature_conditioning --render_test --ft_path ./logs/lego_feature_conditioning/200000.tar --testskip 1
python train_nerf_vit.py --config configs/config.txt --expname lego_transformer --model_type transformer --render_test --ft_path ./logs/lego_transformer/200000.tar --testskip 1