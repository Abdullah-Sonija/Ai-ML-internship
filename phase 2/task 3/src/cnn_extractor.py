"""
cnn_extractor.py
----------------
MobileNetV2-based image feature extractor (PyTorch).

Architecture:
  Input image (3 × 128 × 128)
      └─► MobileNetV2 backbone (frozen, pretrained on ImageNet)
      └─► AdaptiveAvgPool2d(1)
      └─► Flatten
      └─► 1280-d feature vector

Only the 1280-d vectors are returned; the classifier head is discarded.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# ── Constants 

IMAGE_SIZE   = 128
FEATURE_DIM  = 1280           # MobileNetV2 output channels
BATCH_SIZE   = 16             # safe for GT 1030 (2 GB VRAM)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Device 

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[get_device] Using device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1e9
        print(f"  GPU: {props.name}  |  VRAM: {vram:.1f} GB")
    return device


# ── Transforms

def get_transforms(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Model

class MobileNetExtractor(nn.Module):
   
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features           # (B, 1280, H', W')
        self.pool     = nn.AdaptiveAvgPool2d(1) # (B, 1280, 1, 1)

        # Freeze all backbone parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  →  out: (B, 1280)"""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


def load_extractor(device: torch.device) -> MobileNetExtractor:
    model = MobileNetExtractor().to(device)
    model.eval()
    print(f"[load_extractor] MobileNetV2 extractor ready (frozen backbone)")
    return model


# ── Dataset wrapper for extraction 

class ImagePathDataset(Dataset):
    def __init__(self, paths: list, transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"[ImagePathDataset] WARNING: could not load '{path}': {e}")
            # Return a black image as fallback
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


# ── Extraction 

@torch.no_grad()
def extract_features(image_paths: list,
                      model: MobileNetExtractor,
                      transform,
                      device: torch.device,
                      batch_size: int = BATCH_SIZE) -> np.ndarray:
   
    dataset    = ImagePathDataset(image_paths, transform)
    loader     = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,     # keep 0 for Windows compatibility
                            pin_memory=(device.type == "cuda"))

    all_features = []
    model.eval()

    for batch in tqdm(loader, desc="Extracting CNN features"):
        batch = batch.to(device)
        feats = model(batch)                   # (B, 1280)
        all_features.append(feats.cpu().numpy())

    features = np.vstack(all_features).astype(np.float32)
    print(f"[extract_features] Extracted features shape: {features.shape}")
    return features


# ── Persistence ───────────────────────────────────────────────────────────────

def save_features(features: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, features)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"[save_features] Saved {features.shape} → '{save_path}'  ({size_mb:.2f} MB)")


def load_features(save_path: str) -> np.ndarray:
    features = np.load(save_path)
    print(f"[load_features] Loaded features shape: {features.shape} from '{save_path}'")
    return features
