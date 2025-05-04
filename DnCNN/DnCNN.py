import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Define DnCNN Model (grayscale) ===
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = [nn.Conv2d(channels, features, kernel_size, padding=padding), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size, padding=padding),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ]
        layers += [nn.Conv2d(features, channels, kernel_size, padding=padding)]
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # residual learning

# === Setup ===
os.makedirs("output", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = torch.load("dncnn_gray.pth", map_location=device, weights_only=False)
model.eval()
model = model.to(device)

# === Load image ===
img = Image.open("input/testing1.jpg").convert("L")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
clean = transform(img).unsqueeze(0).to(device)  # shape: (1, 1, H, W)

# === Add Gaussian noise ===
sigma = 25 / 255.0
noise = torch.randn_like(clean) * sigma
noisy = torch.clamp(clean + noise, 0.0, 1.0)

# === Denoise ===
start_time = time.time()
with torch.no_grad():
    denoised = model(noisy)
runtime = time.time() - start_time

# === Metrics ===
def to_np(t): return t.squeeze().cpu().numpy()

clean_np = to_np(clean)
denoised_np = to_np(denoised)

psnr_val = psnr(clean_np, denoised_np, data_range=1.0)
ssim_val = ssim(clean_np, denoised_np, data_range=1.0)

# === Save Outputs ===
save_image(clean,    "output/original.png")
save_image(noisy,    "output/noisy.png")
save_image(denoised, "output/denoised.png")

# === Results ===
print(f"✅ PSNR: {psnr_val:.2f} dB")
print(f"✅ SSIM: {ssim_val:.4f}")
print(f"⏱️ Runtime: {runtime:.2f} seconds")
