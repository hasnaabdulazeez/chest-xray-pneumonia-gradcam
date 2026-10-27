"""Grad-CAM (Selvaraju et al., 2017) for any CNN with a convolutional target layer.

The heatmap weights the target layer's activation maps by the spatially
averaged gradient of the class score, then applies ReLU. It shows which image
regions pushed the prediction towards the class, which lets a reviewer check
whether the model looks at the lungs or at shortcuts such as text markers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self._activations = None
        self._gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """Return ``(heatmap in [0, 1] at input resolution, logits)`` for a single image batch."""
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0]) if logits.shape[1] > 1 else 0
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), logits.detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0, 1] heatmap onto an RGB uint8 image using the JET colormap."""
    import cv2

    heat = cv2.resize(heatmap.astype(np.float32), (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1]
    return np.uint8(np.clip((1 - alpha) * image_rgb + alpha * colored, 0, 255))
