"""
gradcam.py – Gradient-weighted Class Activation Mapping for GeoTIFF tiles.

Produces side-by-side images: RGB tile | GradCAM heatmap overlay.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Hooks into a target layer and computes gradient-weighted activations."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Parameters
        ----------
        x         : (1, C, H, W) input tensor on the correct device
        class_idx : class to explain (None → argmax / predicted class)

        Returns
        -------
        cam : (H, W) heatmap in [0, 1]
        """
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def save_gradcam_figure(
    tile: torch.Tensor,          # (C, H, W) float32 in [0,1]
    cam: np.ndarray,             # (H, W)
    true_label: str,
    pred_label: str,
    save_path: str,
    alpha: float = 0.45,
):
    """Save a 2-panel figure: RGB tile | heatmap overlay."""
    # Use first 3 bands (or pad to 3) as the display image
    c = tile.shape[0]
    if c >= 3:
        rgb = tile[:3].permute(1, 2, 0).numpy()
    else:
        rgb = tile[0].numpy()
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    heatmap = cm.jet(cam)[..., :3]                 # (H, W, 3)
    overlay = np.clip(rgb + alpha * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"True: {true_label}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    correct = "✓" if true_label == pred_label else "✗"
    axes[1].set_title(f"Pred: {pred_label}  {correct}", fontsize=11)
    axes[1].axis("off")

    fig.suptitle("GradCAM", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
