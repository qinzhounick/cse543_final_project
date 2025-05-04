import os
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Setup ===
os.makedirs("output", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load DnCNN model ===
model = torch.hub.load("cszn/DnCNN", "dncnn", pretrained=True)
model = model.to(device).eval()

# === Load grayscale image ===
img = Image.open("input/testing1.jpg").convert("L")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # (1, H, W) in [0, 1]
])
clean = transform(img).unsqueeze(0).to(device)  # shape: (1, 1, H, W)

# === Add Gaussian noise ===
sigma = 25 / 255.0  # noise level (25 standard deviation)
noise = torch.randn_like(clean) * sigma
noisy = clean + noise
noisy = torch.clamp(noisy, 0.0, 1.0)

# === Denoising ===
start_time = time.time()
with torch.no_grad():
    denoised = model(noisy)
runtime = time.time() - start_time

# === Convert to NumPy for metrics ===
def to_np(t): return t.squeeze().cpu().numpy()

clean_np = to_np(clean)
noisy_np = to_np(noisy)
denoised_np = to_np(denoised)

# === Metrics ===
psnr_val = psnr(clean_np, denoised_np, data_range=1.0)
ssim_val = ssim(clean_np, denoised_np, data_range=1.0)

# === Save Images ===
save_image(clean,    "output/original.png")
save_image(noisy,    "output/noisy.png")
save_image(denoised, "output/denoised.png")

# === Print Results ===
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
print(f"Runtime: {runtime:.2f} seconds")
