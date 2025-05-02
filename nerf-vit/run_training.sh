#!/bin/bash

python train_nerf_vit.py --config configs/config.txt --expname lego_base --model_type base
python train_nerf_vit.py --config configs/config.txt --expname lego_feature_conditioning --model_type feature_conditioning
python train_nerf_vit.py --config configs/config.txt --expname lego_transformer --model_type transformer