"""
Demo script showing how to use the Grad-CAM + visualization utilities.

This version uses a random "fake" CAM so you can run it without a model.
Replace the fake CAM with a real one from GradCAM(...) in your project.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from utils_vis import show_heatmap_only, show_triplet, MEAN, STD

# Point to the generated pokemon-like images
IMG_DIR = Path("data/pokemon_pics")

def load_random_pokemon_image():
    pngs = list(IMG_DIR.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No images found in {IMG_DIR}")
    path = random.choice(pngs)
    img = Image.open(path).convert("RGB")
    return img, path.name

def main():
    img, name = load_random_pokemon_image()

    # Same kind of transforms you might use for your CNN
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    img_tensor = transform(img)  # (3, H, W)

    # --- Fake CAM for demo: replace this with real Grad-CAM output ---
    # Example: from gradcam import GradCAM, get_last_conv_layer
    # model = ...  # load your model
    # target_layer = get_last_conv_layer(model)
    # gradcam = GradCAM(model, target_layer)
    # cam = gradcam(img_tensor.unsqueeze(0).to(device), target_class=pred_class)
    H, W = 14, 14
    cam = np.random.rand(H, W).astype("float32")
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    # ----------------------------------------------------------------

    show_heatmap_only(cam, title=f"Grad-CAM Heatmap: {name}")
    show_triplet(img_tensor, cam, title=f"Demo (fake CAM) - {name}", alpha=0.5,
                 mean=MEAN, std=STD)

if __name__ == "__main__":
    main()
