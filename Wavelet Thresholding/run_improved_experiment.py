import os
import sys
from improved_wavelet_denoising import experiment_multiple_methods

# Path to your uploaded image
IMAGE_PATH = "Test_Image.jpg"  # Using the exact name from your error log

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)
os.makedirs('results/denoising_comparison', exist_ok=True)

if __name__ == "__main__":
    # Check if the image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file {IMAGE_PATH} not found!")
        print("Please make sure the image file is in the current directory.")
        sys.exit(1)
    
    print(f"Running improved denoising comparison on {IMAGE_PATH}")
    print("This will test wavelet denoising against standard methods")
    print("This may take a moment to complete...")
    
    # Run the experiment
    experiment_multiple_methods(IMAGE_PATH, save_results=True)
    
    print("\nExperiment completed!")
    print("Results saved to 'results/denoising_comparison' directory:")
    print("  - denoising_results.csv: Complete results table")
    print("  - comparison_sigma*.png: Visual comparisons of all methods")
    print("  - psnr_chart_sigma*.png: PSNR comparison charts")
    print("  - ssim_chart_sigma*.png: SSIM comparison charts")
    print("  - Individual denoised images for each method")