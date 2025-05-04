run python DnCNN.py

pip install -r requirements.txt

PSNR and SSIM scores are printed to evaluate denoising quality.

Denoised images are saved in the output/ folder:

    original.png — resized ground truth image

    noisy.png — corrupted image with Gaussian noise

    denoised.png — output from the DnCNN model