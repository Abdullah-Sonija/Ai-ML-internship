"""
Handles all image-side data operations:
  - Collecting image paths from a directory
  - Sampling exactly N images
  - Feature-based pairing (OverallQual → image bucket)
  - Loading and resizing images
  - Visualizing paired samples
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ── Constants 

SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIZE    = (128, 128)
QUAL_MIN        = 1
QUAL_MAX        = 10
N_IMAGES        = 100


# ── Path Collection 

def collect_image_paths(image_dir: str,
                         recursive: bool = True) -> list:
    """
    Walk `image_dir` and return all image file paths.
    Set `recursive=False` to search only the top-level folder.
    """
    paths = []
    if recursive:
        for root, _, files in os.walk(image_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                    paths.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(image_dir)):
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                paths.append(os.path.join(image_dir, f))
    print(f"[collect_image_paths] Found {len(paths)} images in '{image_dir}'")
    return paths


def sample_images(paths: list,
                  n: int = N_IMAGES,
                  seed: int = 42) -> list:
    
    if len(paths) < n:
        raise ValueError(
            f"[sample_images] Only {len(paths)} images found, need at least {n}. "
            "Check your image directory."
        )
    random.seed(seed)
    sampled = random.sample(paths, n)
    print(f"[sample_images] Sampled {n} images (seed={seed})")
    return sorted(sampled)          # sort for reproducibility


# ── Feature-Based Pairing 

def build_qual_buckets(image_paths: list,
                        n_buckets: int = QUAL_MAX) -> dict:
    
    n = len(image_paths)
    per_bucket = n // n_buckets
    remainder  = n % n_buckets

    buckets = {}
    idx = 0
    for q in range(QUAL_MIN, QUAL_MAX + 1):
        # Distribute remainder images to the first `remainder` buckets
        size = per_bucket + (1 if (q - 1) < remainder else 0)
        buckets[q] = image_paths[idx: idx + size]
        idx += size

    sizes = {q: len(v) for q, v in buckets.items()}
    print(f"[build_qual_buckets] Bucket sizes: {sizes}")
    return buckets


def pair_images_to_rows(qual_values: np.ndarray,
                         buckets: dict) -> np.ndarray:
    
    counters  = {q: 0 for q in buckets}
    assigned  = []

    for q in qual_values:
        q = int(np.clip(q, QUAL_MIN, QUAL_MAX))
        imgs    = buckets[q]
        idx     = counters[q] % len(imgs)
        assigned.append(imgs[idx])
        counters[q] += 1

    assigned = np.array(assigned)
    print(f"[pair_images_to_rows] Paired {len(assigned)} rows to images")
    return assigned


def create_pairing_df(qual_values: np.ndarray,
                       image_paths: list,
                       n_images: int = N_IMAGES) -> pd.DataFrame:
   
    sampled = sample_images(image_paths, n=n_images)
    buckets = build_qual_buckets(sampled)
    assigned = pair_images_to_rows(qual_values, buckets)

    df_pair = pd.DataFrame({
        "row_idx":     np.arange(len(qual_values)),
        "OverallQual": qual_values,
        "image_path":  assigned,
    })
    return df_pair


# ── Image Loading 

def load_image(path: str,
               size: tuple = DEFAULT_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img


def load_image_as_array(path: str,
                         size: tuple = DEFAULT_SIZE) -> np.ndarray:
    img = load_image(path, size)
    return np.array(img, dtype=np.float32) / 255.0


# ── Visualization 

def visualize_pairing_samples(df_pair: pd.DataFrame,
                               n_samples: int = 5,
                               size: tuple = DEFAULT_SIZE,
                               save_path: str = None) -> None:
    sample = df_pair.sample(n=min(n_samples, len(df_pair)), random_state=42)

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sample.iterrows()):
        try:
            img = load_image(row["image_path"], size)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"Row {row['row_idx']}\nQual={row['OverallQual']}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample Image–Tabular Pairings (Feature-Based by OverallQual)",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"[visualize_pairing_samples] Saved figure to '{save_path}'")
    plt.show()


def visualize_qual_distribution(df_pair: pd.DataFrame,
                                 save_path: str = None) -> None:
    import seaborn as sns
    counts = df_pair["OverallQual"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
    ax.set_xlabel("OverallQual")
    ax.set_ylabel("Number of rows")
    ax.set_title("Row distribution per OverallQual level")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"[visualize_qual_distribution] Saved figure to '{save_path}'")
    plt.show()
