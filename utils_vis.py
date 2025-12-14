import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Default ImageNet stats; change to your dataset's mean/std if needed.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def denormalize(img_tensor: torch.Tensor, mean=MEAN, std=STD):
    """
    Undo normalization and clamp to [0,1].
    img_tensor: (3, H, W)
    """
    device = img_tensor.device
    mean = torch.tensor(mean, device=device).view(3, 1, 1)
    std = torch.tensor(std, device=device).view(3, 1, 1)
    img = img_tensor * std + mean
    return torch.clamp(img, 0.0, 1.0)

def show_heatmap_only(cam, title="Grad-CAM Heatmap"):
    """
    Show a Grad-CAM heatmap by itself.
    cam: (H', W') numpy array in [0,1]
    """
    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show_triplet(img_tensor, cam, title=None, alpha=0.5, mean=MEAN, std=STD):
    """
    Side-by-side: original, heatmap, overlay.
    img_tensor: (3, H, W), normalized
    cam: (H', W') numpy array in [0,1]
    """
    img = denormalize(img_tensor, mean=mean, std=std).permute(1, 2, 0).cpu().numpy()
    H, W, _ = img.shape

    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = alpha * heatmap + (1.0 - alpha) * img
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title("Heatmap")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Overlay")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
