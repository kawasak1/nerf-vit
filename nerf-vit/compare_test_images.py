import os
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from PIL import Image
import torch
import matplotlib.pyplot as plt
import re

def calculate_metrics():
    """Calculate PSNR, SSIM and LPIPS metrics for all model types"""
    results = {}
    
    # Model types based on the folder names
    model_types = ["base", "feature_conditioning", "transformer"]
    
    # Initialize LPIPS model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # Ground truth directory
    gt_dir = "data/lego/test"
    
    # Get ground truth images (only RGB, not depth)
    gt_images = [f for f in os.listdir(gt_dir) if f.endswith(".png") and not f.endswith("_depth_0001.png")]
    
    # Extract indices from ground truth images using regex
    gt_indices = {}
    for img in gt_images:
        match = re.match(r'r_(\d+)\.png', img)
        if match:
            idx = int(match.group(1))
            gt_indices[idx] = img
    
    for model_type in model_types:
        # Test images directory
        test_dir = f"nerf-vit/test_images/test_{model_type}"
        
        # Get all test images
        test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
        
        # Initialize metrics
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        # Compare each test image with corresponding reference image
        for test_img_file in test_images:
            # Parse index from filename (000.png, 001.png, etc.)
            try:
                test_idx = int(test_img_file.split('.')[0])
            except ValueError:
                print(f"Warning: Could not parse index from test image filename: {test_img_file}")
                continue
            
            # Find corresponding reference image (r_0.png, r_1.png, etc.)
            if test_idx not in gt_indices:
                print(f"Warning: No reference image found for index {test_idx} (test image {test_img_file})")
                continue
                
            gt_img_file = gt_indices[test_idx]
            
            # Load test image
            test_img_path = os.path.join(test_dir, test_img_file)
            test_img = np.array(Image.open(test_img_path).convert('RGB'))
            
            # Load ground truth image
            gt_img_path = os.path.join(gt_dir, gt_img_file)
            gt_img = np.array(Image.open(gt_img_path).convert('RGB'))
            
            # Resize ground truth to match test image dimensions if needed
            if test_img.shape != gt_img.shape:
                gt_img_pil = Image.open(gt_img_path).convert('RGB')
                gt_img_pil = gt_img_pil.resize((test_img.shape[1], test_img.shape[0]), Image.LANCZOS)
                gt_img = np.array(gt_img_pil)
            
            # Calculate PSNR
            psnr_value = psnr(gt_img, test_img)
            psnr_values.append(psnr_value)
            
            # Calculate SSIM
            ssim_value = ssim(gt_img, test_img, channel_axis=2, data_range=255)
            ssim_values.append(ssim_value)
            
            # Calculate LPIPS
            test_tensor = torch.from_numpy(test_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Move tensors to the same device as the LPIPS model
            test_tensor = test_tensor.to(device)
            gt_tensor = gt_tensor.to(device)
            
            with torch.no_grad():
                lpips_value = loss_fn_alex(test_tensor, gt_tensor).item()
            lpips_values.append(lpips_value)
            
            print(f"Model: {model_type}, Image: {test_img_file}, GT: {gt_img_file}")
            print(f"  PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")
        
        # Store results
        results[model_type] = {
            'num_test_images': len(psnr_values),
            'psnr_values': psnr_values,
            'avg_psnr': np.mean(psnr_values) if psnr_values else 0,
            'ssim_values': ssim_values,
            'avg_ssim': np.mean(ssim_values) if ssim_values else 0,
            'lpips_values': lpips_values,
            'avg_lpips': np.mean(lpips_values) if lpips_values else 0
        }
    
    return results

def plot_comparison(results):
    """Create plots comparing the different model types"""
    # Create output directory
    output_dir = "nerf-vit/comparison_results"
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
        plt.figure(figsize=(10, 6))
        values = [results[m][metric_key] for m in model_types]
        bars = plt.bar(model_types, values)
        
        # Add value labels on top of each bar
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{value:.4f}", ha='center', va='bottom')
        
        plt.title(metric_title)
        plt.ylabel(metric_title)
        plt.xlabel("Model Type")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    # Write comparison summary to file
    with open(os.path.join(output_dir, "metrics_summary.txt"), 'w') as f:
        f.write("# NeRF Model Comparison\n\n")
        
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model Type | Average PSNR (dB) | Average SSIM | Average LPIPS |\n")
        f.write("|------------|------------------|--------------|---------------|\n")
        
        for model_type in model_types:
            avg_psnr = results[model_type]['avg_psnr']
            avg_ssim = results[model_type]['avg_ssim']
            avg_lpips = results[model_type]['avg_lpips']
            
            f.write(f"| {model_type} | {avg_psnr:.2f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")
        
        # Find best model for each metric
        best_psnr_model = max(model_types, key=lambda m: results[m]['avg_psnr'])
        f.write(f"\n\nBased on PSNR measurements, the **{best_psnr_model}** approach performs best with {results[best_psnr_model]['avg_psnr']:.2f} dB.\n\n")
        
        best_ssim_model = max(model_types, key=lambda m: results[m]['avg_ssim'])
        f.write(f"Based on SSIM measurements, the **{best_ssim_model}** approach performs best with {results[best_ssim_model]['avg_ssim']:.4f}.\n\n")
        
        best_lpips_model = min(model_types, key=lambda m: results[m]['avg_lpips'])
        f.write(f"Based on LPIPS measurements, the **{best_lpips_model}** approach performs best with {results[best_lpips_model]['avg_lpips']:.4f} (lower is better).\n")
    
    print(f"Comparison results saved to {output_dir}/metrics_summary.txt")

def save_results_to_json(results):
    """Save metrics to a JSON file"""
    output_dir = "nerf-vit/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean data to make it JSON serializable
    for model in results:
        results[model]['psnr_values'] = [float(x) for x in results[model]['psnr_values']]
        results[model]['ssim_values'] = [float(x) for x in results[model]['ssim_values']]
        results[model]['lpips_values'] = [float(x) for x in results[model]['lpips_values']]
    
    with open(os.path.join(output_dir, "metrics_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

def main():
    print("Calculating metrics for all model types...")
    results = calculate_metrics()
    
    print("\nSaving results...")
    save_results_to_json(results)
    
    print("\nPlotting comparison results...")
    plot_comparison(results)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 