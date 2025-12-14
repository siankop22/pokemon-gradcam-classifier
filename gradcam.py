import torch
import torch.nn.functional as F
import torch.nn as nn

class GradCAM:
    """
    Minimal Grad-CAM implementation for a PyTorch CNN model.
    Uses torch.autograd.grad for more reliable gradient computation.

    Usage:
        target_layer = get_last_conv_layer(model)  # or pick manually
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(input_tensor, target_class=predicted_class)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: (1, C, H, W) tensor on same device as model
        target_class: index of class to visualize. If None, uses argmax.
        Returns:
            cam: (H', W') numpy array in [0, 1]
        """
        self.model.zero_grad()
        
        # Forward pass - we need to capture activations
        # Hook to capture activations
        activations = None
        def forward_hook(module, inputs, output):
            nonlocal activations
            activations = output
        
        handle = self.target_layer.register_forward_hook(forward_hook)
        
        try:
            output = self.model(input_tensor)  # (1, num_classes)

            if target_class is None:
                target_class = int(output.argmax(dim=1).item())

            score = output[0, target_class]
            
            # Use autograd.grad instead of backward() to avoid hook issues
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=activations,
                retain_graph=False,
                create_graph=False
            )[0]
            
            # Channel-wise weights via global average pooling on gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted sum of activations
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

            cam = F.relu(cam)
            cam = cam.squeeze()  # (H', W')

            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

            return cam.detach().cpu().numpy()
        finally:
            handle.remove()


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Utility to automatically grab the last Conv2d layer from a model.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in the model.")
    return last_conv
