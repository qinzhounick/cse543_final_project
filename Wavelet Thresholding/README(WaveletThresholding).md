# Wavelet Thresholding for Image Denoising

This repository contains an implementation of image denoising using wavelet thresholding techniques. The algorithm leverages the powerful multi-resolution analysis capabilities of wavelet transforms to effectively remove noise while preserving important image details.

## Overview

Wavelet-based denoising works by:
1. Decomposing the noisy image into wavelet coefficients
2. Applying thresholding to the detail coefficients to suppress noise
3. Reconstructing the denoised image using the modified coefficients

The implementation supports various wavelet families, decomposition levels, and thresholding techniques.

## Features

- **Multiple wavelet families** support (`db8`, `sym8`, etc.)
- **Adaptive thresholding** using noise level estimation
- **Soft and hard thresholding** modes
- **Multi-level decomposition** options
- **Threshold scaling** for fine-tuning denoising strength
- **Color and grayscale image** support
- **Comprehensive metrics** (PSNR, SSIM) for result evaluation
- **Performance measurement** with execution time tracking
- **Visualization tools** for comparing results

## Requirements

- Python 3.6+
- NumPy
- OpenCV (cv2)
- Matplotlib
- PyWavelets (pywt)
- scikit-image
- pandas (optional, for metrics CSV export)

Install required packages using:

```bash
pip install numpy opencv-python matplotlib pywavelets scikit-image pandas
```

## Usage

### Basic Usage

```bash
python run_wavelet_denoising.py --image path/to/your/image.jpg
```

### Advanced Options

```bash
python run_wavelet_denoising.py --image path/to/your/image.jpg --noise 15 --no-add-noise --no-save
```

### Command Line Arguments

- `--image`: Path to the input image (required)
- `--noise`: Noise level for synthetic noise (default: 25)
- `--no-save`: Do not save results and figures
- `--no-add-noise`: Do not add synthetic noise (use if image is already noisy)

### Using as a Module

You can also import and use the denoising function directly in your code:

```python
from wavelet_denoising import wavelet_denoising, run_wavelet_denoising

# For a single denoising operation with specific parameters
denoised_image, execution_time = wavelet_denoising(
    noisy_image, 
    wavelet='db8', 
    level=3, 
    threshold_mode='soft', 
    threshold_scale=1.0
)

# For full parameter sweep and evaluation
run_wavelet_denoising(
    'path/to/image.jpg', 
    noise_sigma=25, 
    save_results=True, 
    add_noise=True
)
```

## Code Structure

- `wavelet_denoising.py`: Main implementation of the wavelet thresholding algorithm
- `run_wavelet_denoising.py`: Script to run the denoising with various parameters

## Algorithm Details

### Noise Estimation

The implementation includes an adaptive noise estimation function that uses median absolute deviation (MAD) to estimate the noise level in the image, which is then used to calculate an appropriate threshold value.

### Thresholding Approaches

- **Soft Thresholding**: Shrinks coefficients toward zero (better for preserving smooth features)
- **Hard Thresholding**: Sets coefficients below threshold to zero (better for preserving edges)

### Parameter Optimization

The code automatically tests multiple parameter combinations:
- Wavelet families: `db8`, `sym8`
- Decomposition levels: 2, 3
- Threshold modes: soft, hard
- Threshold scales: 0.5, 1.0

## Output

The results are saved in the `results` directory:

- `wavelet_denoising_metrics.csv`: CSV file with PSNR, SSIM, and runtime metrics for all parameter combinations
- `original_vs_noisy.png`: Comparison of original and noisy images
- `best_denoised.png`: Side-by-side comparison of best denoising results
- Individual denoised images for each parameter combination

## Example Results

When running the code, you'll see detailed information about:
- Noise estimation for each image channel
- Applied threshold values
- PSNR and SSIM metrics for each parameter combination
- Execution time
- Best parameter sets based on PSNR and SSIM

## Customization

To experiment with additional wavelet families or parameters, modify the parameter lists in the `run_wavelet_denoising` function:

```python
wavelet_families = ['db8', 'sym8', 'haar', 'coif3']  # Add more wavelets
threshold_modes = ['soft', 'hard']
levels = [2, 3, 4]  # Add deeper decomposition levels
scales = [0.5, 0.75, 1.0, 1.25]  # Add more threshold scales
```

## Performance Considerations

- Higher decomposition levels require more computation time but might provide better denoising for certain images
- Color images are processed channel by channel, which takes about 3x longer than grayscale images
- The `db8` wavelet generally provides good results for natural images but may not be optimal for all image types

## License

This code is provided for educational and research purposes.

## References

- Donoho, D.L., & Johnstone, I.M. (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika, 81(3), 425-455.
- Chang, S.G., Yu, B., & Vetterli, M. (2000). Adaptive wavelet thresholding for image denoising and compression. IEEE Transactions on Image Processing, 9(9), 1532-1546.
