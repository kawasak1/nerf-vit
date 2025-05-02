import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import lpips
from PIL import Image
import torch

def run_model_training(model_type):
    """Run training for the specified model type"""
    command = f"python train_nerf_vit.py --config configs/config.txt --expname nerf_vit_{model_type} --model_type {model_type}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

def run_test_evaluation(model_type):
    """Run test evaluation for the specified model type"""
    command = f"python train_nerf_vit.py --config configs/config.txt --expname nerf_vit_{model_type} --model_type {model_type} --render_only --render_test"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

def calculate_metrics(model_types):
    """Calculate and compare metrics across all model types"""
    results = {}
    
    # Initialize LPIPS model
    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find ground truth directory
    config_file = "configs/config.txt"
    datadir = None
    with open(config_file, 'r') as f:
        for line in f:
            if line.strip().startswith('datadir'):
                datadir = line.strip().split('=')[1].strip()
                break
    
    if not datadir:
        print("Warning: Could not find datadir in config, SSIM and LPIPS metrics will not be calculated")
    
    for model_type in model_types:
        expname = f"nerf_vit_{model_type}"
        base_dir = f"logs/{expname}"
        
        # Find test render directory
        render_dirs = [d for d in os.listdir(base_dir) if d.startswith("renderonly_test")]
        if not render_dirs:
            print(f"No render directories found for {model_type}")
            continue
        
        latest_render_dir = os.path.join(base_dir, render_dirs[-1])
        
        # Get all PNG files with test renderings
        test_images = sorted([f for f in os.listdir(latest_render_dir) if f.endswith(".png")])
        
        # Initialize metrics
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        # Read the PSNR values from the log file if it exists
        log_file = os.path.join(latest_render_dir, "test_metrics.json")
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                metrics = json.load(f)
                psnr_values = metrics.get('psnr', [])
        
        # Calculate SSIM and LPIPS if datadir is available
        if datadir and len(test_images) > 0:
            # Load ground truth images
            gt_dir = os.path.join(datadir, "test")
            gt_images = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
            
            # Calculate SSIM and LPIPS for each test image
            for i, img_file in enumerate(test_images):
                if i >= len(gt_images):
                    break
                    
                # Load predicted image
                pred_img_path = os.path.join(latest_render_dir, img_file)
                pred_img = np.array(Image.open(pred_img_path).convert('RGB'))
                
                # Load ground truth image
                gt_img_path = os.path.join(gt_dir, gt_images[i])
                gt_img = np.array(Image.open(gt_img_path).convert('RGB'))
                
                # Calculate SSIM
                ssim_value = ssim(gt_img, pred_img, channel_axis=2, data_range=255)
                ssim_values.append(ssim_value)
                
                # Calculate LPIPS
                pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                # Move tensors to the same device as the LPIPS model
                device = next(loss_fn_alex.parameters()).device
                pred_tensor = pred_tensor.to(device)
                gt_tensor = gt_tensor.to(device)
                
                with torch.no_grad():
                    lpips_value = loss_fn_alex(pred_tensor, gt_tensor).item()
                lpips_values.append(lpips_value)
        
        # Store results
        results[model_type] = {
            'num_test_images': len(test_images),
            'psnr_values': psnr_values,
            'avg_psnr': np.mean(psnr_values) if psnr_values and isinstance(psnr_values[0], (int, float)) else "N/A",
            'ssim_values': ssim_values,
            'avg_ssim': np.mean(ssim_values) if ssim_values else "N/A",
            'lpips_values': lpips_values,
            'avg_lpips': np.mean(lpips_values) if lpips_values else "N/A",
            'render_dir': latest_render_dir
        }
    
    return results

def plot_comparison(results):
    """Create plots comparing the different model types"""
    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model types with available results
    model_types = list(results.keys())
    if not model_types:
        print("No results to plot")
        return
    
    # Create bar charts for different metrics
    metrics = [
        ('avg_psnr', 'Average PSNR (dB)', 'psnr_comparison.png', True),
        ('avg_ssim', 'Average SSIM', 'ssim_comparison.png', True),
        ('avg_lpips', 'Average LPIPS (lower is better)', 'lpips_comparison.png', False)
    ]
    
    for metric_key, metric_title, filename, higher_is_better in metrics:
        values = [results[m][metric_key] for m in model_types]
        if all(isinstance(v, (int, float)) for v in values):
            plt.figure(figsize=(10, 6))
            plt.bar(model_types, values)
            plt.title(metric_title)
            plt.ylabel(metric_title)
            plt.xlabel("Model Type")
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
    
    # Write comparison summary to file
    with open(os.path.join(output_dir, "comparison_summary.txt"), 'w') as f:
        f.write("# NeRF-ViT Model Comparison\n\n")
        
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model Type | Average PSNR (dB) | Average SSIM | Average LPIPS |\n")
        f.write("|------------|------------------|--------------|---------------|\n")
        
        for model_type in model_types:
            avg_psnr = results[model_type]['avg_psnr']
            avg_psnr_str = f"{avg_psnr:.2f}" if isinstance(avg_psnr, (int, float)) else "N/A"
            
            avg_ssim = results[model_type]['avg_ssim']
            avg_ssim_str = f"{avg_ssim:.4f}" if isinstance(avg_ssim, (int, float)) else "N/A"
            
            avg_lpips = results[model_type]['avg_lpips']
            avg_lpips_str = f"{avg_lpips:.4f}" if isinstance(avg_lpips, (int, float)) else "N/A"
            
            f.write(f"| {model_type} | {avg_psnr_str} | {avg_ssim_str} | {avg_lpips_str} |\n")
        
        f.write("\n## Architecture Analysis\n\n")
        f.write("### Feature Conditioning NeRF\n")
        f.write("This approach extracts global image features from the ViT and concatenates them with the input coordinates.\n")
        f.write("- **Pros**: Simple implementation, leverages ViT's global understanding of the scene\n")
        f.write("- **Cons**: May not fully utilize the spatial relations captured by the ViT\n\n")
        
        f.write("### Transformer-based NeRF\n")
        f.write("This approach replaces part of the MLP layers with transformer blocks that attend to ViT features.\n")
        f.write("- **Pros**: Better spatial reasoning through cross-attention mechanisms\n")
        f.write("- **Cons**: More complex architecture and potentially higher computational cost\n\n")
        
        f.write("### Cross-Attention NeRF\n")
        f.write("This approach uses cross-attention to relate 3D points with image features from the ViT.\n")
        f.write("- **Pros**: Direct relationship modeling between 3D points and image features\n")
        f.write("- **Cons**: Attention computation can be memory-intensive\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Find best model for each metric
        if all(isinstance(results[m]['avg_psnr'], (int, float)) for m in model_types):
            best_psnr_model = max(model_types, key=lambda m: results[m]['avg_psnr'])
            f.write(f"Based on PSNR measurements, the **{best_psnr_model}** approach performs best with {results[best_psnr_model]['avg_psnr']:.2f} dB.\n\n")
            
        if all(isinstance(results[m]['avg_ssim'], (int, float)) for m in model_types):
            best_ssim_model = max(model_types, key=lambda m: results[m]['avg_ssim'])
            f.write(f"Based on SSIM measurements, the **{best_ssim_model}** approach performs best with {results[best_ssim_model]['avg_ssim']:.4f}.\n\n")
            
        if all(isinstance(results[m]['avg_lpips'], (int, float)) for m in model_types):
            best_lpips_model = min(model_types, key=lambda m: results[m]['avg_lpips'])
            f.write(f"Based on LPIPS measurements, the **{best_lpips_model}** approach performs best with {results[best_lpips_model]['avg_lpips']:.4f} (lower is better).\n")
        
        if not (all(isinstance(results[m]['avg_psnr'], (int, float)) for m in model_types) or 
                all(isinstance(results[m]['avg_ssim'], (int, float)) for m in model_types) or
                all(isinstance(results[m]['avg_lpips'], (int, float)) for m in model_types)):
            f.write("Incomplete metrics available for full comparison. Please run evaluation on all models to get a complete comparison.\n")
    
    print(f"Comparison results saved to {output_dir}/comparison_summary.txt")

def main():
    parser = argparse.ArgumentParser(description="Train and compare NeRF-ViT models")
    parser.add_argument('--train', action='store_true', help='Train all models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate all models')
    parser.add_argument('--compare', action='store_true', help='Compare model results')
    args = parser.parse_args()
    
    model_types = ["base", "feature_conditioning", "transformer", "cross_attention"]
    
    if args.train:
        for model_type in model_types:
            run_model_training(model_type)
    
    if args.evaluate:
        for model_type in model_types:
            run_test_evaluation(model_type)
    
    if args.compare:
        results = calculate_metrics(model_types)
        plot_comparison(results)
    
    if not (args.train or args.evaluate or args.compare):
        print("No action specified. Use --train, --evaluate, or --compare")

if __name__ == "__main__":
    main() 