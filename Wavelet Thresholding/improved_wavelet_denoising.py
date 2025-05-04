import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import os
import pandas as pd

def add_gaussian_noise(image, sigma=25):
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image (normalized to [0,1])
        sigma: Standard deviation of the Gaussian noise (in range [0,255])
        
    Returns:
        Noisy image
    """
    # Convert sigma from [0,255] to [0,1] range
    sigma_normalized = sigma / 255.0
    
    # Generate noise and add to image
    noise = np.random.normal(0, sigma_normalized, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Print debug info
    print(f"Noise generation - Original image shape: {image.shape}, min: {image.min():.4f}, max: {image.max():.4f}")
    print(f"Noisy image - min: {noisy_image.min():.4f}, max: {noisy_image.max():.4f}")
    
    return noisy_image

def wavelet_denoising(noisy_image, wavelet='db8', level=3, threshold_mode='soft', threshold_scale=1.0):
    """
    Denoise an image using wavelet thresholding.
    
    Args:
        noisy_image: Input noisy image
        wavelet: Wavelet family to use
        level: Decomposition level
        threshold_mode: 'soft' or 'hard' thresholding
        threshold_scale: Scale factor for the threshold (adjust to tune denoising strength)
        
    Returns:
        Denoised image and execution time
    """
    # Start timing
    start_time = time.time()
    
    # Debug input image
    print(f"Input to wavelet_denoising - shape: {noisy_image.shape}, dtype: {noisy_image.dtype}")
    print(f"Input image stats - min: {noisy_image.min():.4f}, max: {noisy_image.max():.4f}")
    
    # Ensure the image is in float format
    if noisy_image.dtype != np.float32:
        noisy_image = noisy_image.astype(np.float32)
        if noisy_image.max() > 1.0:
            noisy_image /= 255.0
        print("Converted image to float32")
    
    # Create a copy to work with (avoid modifying the original)
    img = noisy_image.copy()
    
    # Handle grayscale vs. color images
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("Processing color image")
        # Process each channel separately
        denoised_channels = []
        for i in range(3):
            # Get the current channel
            channel = img[:, :, i]
            
            # Decompose using wavelet transform
            coeffs = pywt.wavedec2(channel, wavelet, level=level)
            
            # Estimate noise using the finest-scale coefficients (HH subband)
            finest_level_coeffs = coeffs[1][2]  # HH subband
            noise_sigma = np.median(np.abs(finest_level_coeffs)) / 0.6745
            
            # Ensure threshold is not too small
            noise_sigma = max(noise_sigma, 0.01)
            
            # Calculate threshold (universal threshold)
            threshold = noise_sigma * np.sqrt(2 * np.log(channel.size)) * threshold_scale
            
            print(f"Channel {i} - MAD: {noise_sigma:.6f}, Threshold: {threshold:.6f}")
            
            # Apply thresholding to detail coefficients
            new_coeffs = list(coeffs)
            for j in range(1, len(coeffs)):
                # Apply threshold to each subband (horizontal, vertical, diagonal)
                if threshold_mode == 'soft':
                    new_coeffs[j] = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs[j])
                else:  # hard thresholding
                    new_coeffs[j] = tuple(pywt.threshold(c, threshold, mode='hard') for c in coeffs[j])
            
            # Reconstruct the channel
            denoised_channel = pywt.waverec2(new_coeffs, wavelet)
            
            # Handle potential size mismatch
            denoised_channel = denoised_channel[:channel.shape[0], :channel.shape[1]]
            
            # Debug info
            print(f"Denoised channel {i} - min: {denoised_channel.min():.4f}, max: {denoised_channel.max():.4f}")
            
            # Add the denoised channel to the list
            denoised_channels.append(denoised_channel)
        
        # Stack channels
        denoised_image = np.stack(denoised_channels, axis=2)
    else:
        print("Processing grayscale image")
        # Process grayscale image
        coeffs = pywt.wavedec2(img, wavelet, level=level)
        
        # Estimate noise using the finest-scale coefficients (HH subband)
        finest_level_coeffs = coeffs[1][2]  # HH subband
        noise_sigma = np.median(np.abs(finest_level_coeffs)) / 0.6745
        
        # Ensure threshold is not too small
        noise_sigma = max(noise_sigma, 0.01)
        
        # Calculate threshold (universal threshold)
        threshold = noise_sigma * np.sqrt(2 * np.log(img.size)) * threshold_scale
        
        print(f"Grayscale - MAD: {noise_sigma:.6f}, Threshold: {threshold:.6f}")
        
        # Apply thresholding to detail coefficients
        new_coeffs = list(coeffs)
        for j in range(1, len(coeffs)):
            # Apply threshold to each subband (horizontal, vertical, diagonal)
            if threshold_mode == 'soft':
                new_coeffs[j] = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs[j])
            else:  # hard thresholding
                new_coeffs[j] = tuple(pywt.threshold(c, threshold, mode='hard') for c in coeffs[j])
        
        # Reconstruct image
        denoised_image = pywt.waverec2(new_coeffs, wavelet)
        
        # Handle potential size mismatch
        denoised_image = denoised_image[:img.shape[0], :img.shape[1]]
    
    # Ensure the output is in the valid range [0, 1]
    denoised_image = np.clip(denoised_image, 0, 1)
    
    # Debug output
    print(f"Final output image - min: {denoised_image.min():.4f}, max: {denoised_image.max():.4f}")
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return denoised_image, execution_time

def denoise_standard(noisy_image, method='gaussian', **kwargs):
    """
    Apply standard denoising methods for comparison.
    
    Args:
        noisy_image: Input noisy image
        method: Denoising method ('gaussian', 'bilateral', 'median')
        kwargs: Additional parameters for the denoising method
        
    Returns:
        Denoised image and execution time
    """
    # Start timing
    start_time = time.time()
    
    # Ensure the image is in uint8 format
    if noisy_image.dtype != np.uint8:
        if np.max(noisy_image) <= 1.0:
            img_uint8 = (noisy_image * 255).astype(np.uint8)
        else:
            img_uint8 = noisy_image.astype(np.uint8)
    else:
        img_uint8 = noisy_image
    
    # Apply denoising method
    if method == 'gaussian':
        # Default parameters if not provided
        ksize = kwargs.get('ksize', (5, 5))
        sigma = kwargs.get('sigma', 1.5)
        
        if len(img_uint8.shape) == 3:  # Color image
            denoised_img = cv2.GaussianBlur(img_uint8, ksize, sigma)
        else:  # Grayscale image
            denoised_img = cv2.GaussianBlur(img_uint8, ksize, sigma)
            
    elif method == 'bilateral':
        # Default parameters if not provided
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        
        if len(img_uint8.shape) == 3:  # Color image
            denoised_img = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
        else:  # Grayscale image
            denoised_img = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
            
    elif method == 'median':
        # Default parameters if not provided
        ksize = kwargs.get('ksize', 5)
        
        if len(img_uint8.shape) == 3:  # Color image
            denoised_img = cv2.medianBlur(img_uint8, ksize)
        else:  # Grayscale image
            denoised_img = cv2.medianBlur(img_uint8, ksize)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert back to float32 [0,1] if input was float
    if noisy_image.dtype == np.float32 or noisy_image.dtype == np.float64:
        denoised_img = denoised_img.astype(np.float32) / 255.0
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return denoised_img, execution_time

def experiment_multiple_methods(image_path, save_results=True):
    """
    Run experiment with several denoising methods for comparison.
    
    Args:
        image_path: Path to the original image
        save_results: Whether to save results and figures
    """
    # Load image
    print(f"Loading image from {image_path}...")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}. Check the file path.")
    
    print(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
    print(f"Original image min: {original_image.min()}, max: {original_image.max()}")
    
    # Convert to RGB (from BGR) if it's a color image
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        print("Converted from BGR to RGB")
    elif len(original_image.shape) == 2:
        print("Image is grayscale")
    
    # Save original image for reference
    if save_results:
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/denoising_comparison', exist_ok=True)
        cv2.imwrite('results/denoising_comparison/original.png', 
                   cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR) if len(original_image.shape) == 3 
                   else original_image)
    
    # Normalize to [0, 1]
    original_float = original_image.astype(np.float32) / 255.0
    print(f"Normalized image min: {original_float.min()}, max: {original_float.max()}")
    
    # Define noise levels to test
    noise_levels = [25]  # Using only one noise level for simplicity
    
    # Initialize results table
    results_data = []
    
    # Loop through noise levels
    for noise_sigma in noise_levels:
        print(f"\nAdding noise with sigma = {noise_sigma}...")
        
        # Add noise to original image
        noisy_image = add_gaussian_noise(original_float, sigma=noise_sigma)
        
        # Calculate metrics for noisy image
        noisy_psnr_val = psnr(original_float, noisy_image)
        noisy_ssim_val = ssim(original_float, noisy_image, 
                            channel_axis=2 if len(original_float.shape) > 2 else None,
                            data_range=1.0)
        
        print(f"Noisy image (sigma={noise_sigma}) - PSNR: {noisy_psnr_val:.2f} dB, SSIM: {noisy_ssim_val:.4f}")
        
        # Add noisy image to results
        results_data.append({
            'Noise_Level': noise_sigma,
            'Method': 'Noisy',
            'PSNR': noisy_psnr_val,
            'SSIM': noisy_ssim_val,
            'Runtime': 0
        })
        
        # Save noisy image
        if save_results:
            noisy_uint8 = (noisy_image * 255).astype(np.uint8)
            noisy_img_path = f"results/denoising_comparison/noisy_sigma{noise_sigma}.png"
            if len(noisy_image.shape) == 3:
                cv2.imwrite(noisy_img_path, cv2.cvtColor(noisy_uint8, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(noisy_img_path, noisy_uint8)
        
        # Test different denoising methods
        
        # 1. Standard OpenCV methods
        methods = {
            'Gaussian': ('gaussian', {'ksize': (5, 5), 'sigma': 1.5}),
            'Bilateral': ('bilateral', {'d': 9, 'sigma_color': 75, 'sigma_space': 75}),
            'Median': ('median', {'ksize': 5})
        }
        
        for method_name, (method, params) in methods.items():
            print(f"\nTesting {method_name} filter...")
            
            try:
                denoised_image, runtime = denoise_standard(noisy_image, method=method, **params)
                
                # Calculate metrics
                denoised_psnr = psnr(original_float, denoised_image)
                denoised_ssim = ssim(original_float, denoised_image, 
                                  channel_axis=2 if len(original_float.shape) > 2 else None,
                                  data_range=1.0)
                
                # Add to results
                results_data.append({
                    'Noise_Level': noise_sigma,
                    'Method': method_name,
                    'PSNR': denoised_psnr,
                    'SSIM': denoised_ssim,
                    'Runtime': runtime
                })
                
                print(f"  PSNR: {denoised_psnr:.2f} dB, SSIM: {denoised_ssim:.4f}, Runtime: {runtime:.4f} s")
                
                # Save denoised image
                if save_results:
                    denoised_uint8 = (denoised_image * 255).astype(np.uint8)
                    denoised_img_path = f"results/denoising_comparison/denoised_{method_name}_sigma{noise_sigma}.png"
                    if len(denoised_image.shape) == 3:
                        cv2.imwrite(denoised_img_path, cv2.cvtColor(denoised_uint8, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(denoised_img_path, denoised_uint8)
                
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        # 2. Wavelet methods
        wavelet_params = [
            ('db8', 2, 'soft', 0.5),
            ('db8', 2, 'hard', 0.5),
            ('sym8', 2, 'soft', 0.5),
            ('sym8', 2, 'hard', 0.5),
        ]
        
        for wavelet, level, mode, scale in wavelet_params:
            method_name = f"Wavelet_{wavelet}_L{level}_{mode}_s{scale}"
            print(f"\nTesting {method_name}...")
            
            try:
                denoised_image, runtime = wavelet_denoising(
                    noisy_image, wavelet=wavelet, level=level, 
                    threshold_mode=mode, threshold_scale=scale)
                
                # Calculate metrics
                denoised_psnr = psnr(original_float, denoised_image)
                denoised_ssim = ssim(original_float, denoised_image, 
                                  channel_axis=2 if len(original_float.shape) > 2 else None,
                                  data_range=1.0)
                
                # Add to results
                results_data.append({
                    'Noise_Level': noise_sigma,
                    'Method': method_name,
                    'PSNR': denoised_psnr,
                    'SSIM': denoised_ssim,
                    'Runtime': runtime
                })
                
                print(f"  PSNR: {denoised_psnr:.2f} dB, SSIM: {denoised_ssim:.4f}, Runtime: {runtime:.4f} s")
                
                # Save denoised image
                if save_results:
                    denoised_uint8 = (denoised_image * 255).astype(np.uint8)
                    denoised_img_path = f"results/denoising_comparison/denoised_{method_name}_sigma{noise_sigma}.png"
                    if len(denoised_image.shape) == 3:
                        cv2.imwrite(denoised_img_path, cv2.cvtColor(denoised_uint8, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(denoised_img_path, denoised_uint8)
                
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    # Convert results to DataFrame and save to CSV
    if save_results and results_data:
        df = pd.DataFrame(results_data)
        df.to_csv('results/denoising_comparison/denoising_results.csv', index=False)
        print(f"Saved results to results/denoising_comparison/denoising_results.csv")
    
    # Create comparison visualization
    if save_results and results_data:
        create_comparison_visualization(results_data, noise_levels, image_path)
    
    return results_data

def create_comparison_visualization(results_data, noise_levels, image_path):
    """
    Create a visual comparison of all denoising methods.
    """
    # Get original image filename without extension
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # For each noise level, create a separate figure
    for noise_sigma in noise_levels:
        # Filter results for this noise level
        noise_results = [r for r in results_data if r['Noise_Level'] == noise_sigma]
        
        if not noise_results:
            continue
        
        # Sort methods by PSNR (descending)
        methods_sorted = sorted(noise_results, key=lambda x: x['PSNR'], reverse=True)
        
        # Create a grid figure with all methods
        n_methods = len(methods_sorted)
        n_cols = min(4, n_methods)  # Max 4 columns
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Add each method to the figure
        for i, result in enumerate(methods_sorted):
            row = i // n_cols
            col = i % n_cols
            
            method_name = result['Method']
            
            # Get the image path
            if method_name == 'Noisy':
                img_path = f'results/denoising_comparison/noisy_sigma{noise_sigma}.png'
            else:
                img_path = f'results/denoising_comparison/denoised_{method_name}_sigma{noise_sigma}.png'
            
            # Load and display the image
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{method_name}\nPSNR: {result['PSNR']:.2f} dB\nSSIM: {result['SSIM']:.4f}")
            else:
                axes[row, col].text(0.5, 0.5, f"Image not available", 
                                   horizontalalignment='center', verticalalignment='center')
                axes[row, col].set_title(method_name)
            
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_methods, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        # Add overall title
        plt.suptitle(f'Denoising Methods Comparison: {img_name} with Noise σ={noise_sigma}', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        plt.savefig(f'results/denoising_comparison/comparison_sigma{noise_sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar charts for PSNR and SSIM
        plt.figure(figsize=(12, 6))
        
        # Extract method names and PSNR values
        methods = [r['Method'] for r in methods_sorted]
        psnr_values = [r['PSNR'] for r in methods_sorted]
        
        # Plot bars
        plt.bar(methods, psnr_values)
        plt.xlabel('Method')
        plt.ylabel('PSNR (dB)')
        plt.title(f'PSNR Comparison of Denoising Methods (Noise σ={noise_sigma})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/denoising_comparison/psnr_chart_sigma{noise_sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar chart for SSIM values
        plt.figure(figsize=(12, 6))
        
        # Extract SSIM values
        ssim_values = [r['SSIM'] for r in methods_sorted]
        
        # Plot bars
        plt.bar(methods, ssim_values)
        plt.xlabel('Method')
        plt.ylabel('SSIM')
        plt.title(f'SSIM Comparison of Denoising Methods (Noise σ={noise_sigma})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/denoising_comparison/ssim_chart_sigma{noise_sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Wavelet Denoising Experiment')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--no-save', action='store_false', dest='save_results', 
                       help='Do not save results and figures')
    
    args = parser.parse_args()
    
    # Run the experiment
    experiment_multiple_methods(args.image, args.save_results)