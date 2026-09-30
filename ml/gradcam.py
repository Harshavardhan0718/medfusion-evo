from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    # Matplotlib-like JET approximation without importing matplotlib.
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, args, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output:
            self.gradients = grad_output[0].detach()

    def __call__(self, original: Image.Image) -> bytes:
        transform = self.model_input_transform()
        x = transform(original).unsqueeze(0)
        self.model.zero_grad(set_to_none=True)

        self.model.eval()
        logits = self.model(x)
        score = logits[:, 0].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=(original.height, original.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        cam -= cam.min()
        denom = cam.max()
        if denom > 1e-8:
            cam /= denom

        heat = (_jet_colormap(cam) * 255).astype(np.uint8)
        base = np.asarray(original.resize((original.width, original.height))).astype(np.float32)
        alpha = (0.45 * cam[..., None]).astype(np.float32)
        blended = base * (1.0 - alpha) + heat.astype(np.float32) * alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out = io.BytesIO()
        Image.fromarray(blended).save(out, format="PNG")
        return out.getvalue()

    def model_input_transform(self):
        # Kept local to avoid sharing mutable transform state.
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def close(self):
        for handle in self._handles:
            handle.remove()
