from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from ml.gradcam import GradCAM

MODEL_REPO = os.getenv("IMAGE_MODEL_REPO", "nismal1u/chestX-rays-DenseNet121")
MODEL_FILE = os.getenv("IMAGE_MODEL_FILE", "ckpt.pt")


def _download_checkpoint() -> str:
    cache_dir = Path(os.getenv("HF_HOME", "/tmp/huggingface"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=str(cache_dir),
    )


def build_model() -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier.in_features, 1),
    )
    return model


def _load_state_dict(model: nn.Module, checkpoint_path: str) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    # Remove optional DataParallel prefix.
    checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)


class ImageModelService:
    def __init__(self):
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        self.device = torch.device("cpu")
        self.model = build_model().to(self.device)
        _load_state_dict(self.model, _download_checkpoint())
        self.model.eval()

        # Must match the model card preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.gradcam = GradCAM(
            model=self.model,
            target_layer=self.model.features.denseblock4.denselayer16.conv2,
        )

    def predict(self, image: Image.Image) -> dict:
        x = self.transform(image).unsqueeze(0).to(self.device)
        # The published model uses MC Dropout to estimate epistemic uncertainty.
        probs = []
        self.model.train()
        with torch.no_grad():
            for _ in range(8):
                logits = self.model(x)
                probs.append(torch.sigmoid(logits[:, 0]).item())
        self.model.eval()

        mean_prob = float(np.mean(probs))
        uncertainty = float(np.std(probs))

        # Deterministic feature extraction and Grad-CAM.
        with torch.no_grad():
            features = self.model.features(x)
            features = torch.relu(features)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            embedding = pooled.flatten(1)[0].cpu().numpy().astype(np.float32)

        gradcam_png = self.gradcam(image)
        return {
            "probability": mean_prob,
            "uncertainty": uncertainty,
            "embedding": embedding,
            "gradcam_png": gradcam_png,
        }
