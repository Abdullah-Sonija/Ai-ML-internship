import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ── Dataset 

class MultimodalDataset(Dataset):
    """
    Each sample contains:
      - tabular_features : 1-D float tensor  (num_tabular_features,)
      - image_features   : 1-D float tensor  (1280,)
      - target           : scalar float tensor
    """

    def __init__(self,
                 tabular_features: np.ndarray,
                 image_features:   np.ndarray,
                 targets:          np.ndarray):
        
        assert len(tabular_features) == len(image_features) == len(targets), (
            "All inputs must have the same number of rows. "
            f"Got tab={len(tabular_features)}, img={len(image_features)}, "
            f"y={len(targets)}"
        )
        self.X_tab  = torch.from_numpy(tabular_features.astype(np.float32))
        self.X_img  = torch.from_numpy(image_features.astype(np.float32))
        self.y      = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X_tab[idx], self.X_img[idx], self.y[idx]


# ── DataLoader Factory 
def create_dataloaders(tabular_features: np.ndarray,
                        image_features:   np.ndarray,
                        targets:          np.ndarray,
                        batch_size:       int   = 32,
                        val_split:        float = 0.15,
                        test_split:       float = 0.10,
                        seed:             int   = 42,
                        num_workers:      int   = 0) -> tuple:

    dataset = MultimodalDataset(tabular_features, image_features, targets)
    N       = len(dataset)

    n_test  = int(N * test_split)
    n_val   = int(N * val_split)
    n_train = N - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    split_sizes = {"train": n_train, "val": n_val, "test": n_test}
    print(f"[create_dataloaders] Split → {split_sizes}  |  batch_size={batch_size}")
    return train_loader, val_loader, test_loader, split_sizes


# ── Utility 

def describe_dataset(tabular_features: np.ndarray,
                      image_features:   np.ndarray,
                      targets:          np.ndarray) -> None:

    print("=" * 50)
    print("  Multimodal Dataset Summary")
    print("=" * 50)
    print(f"  Rows              : {len(targets)}")
    print(f"  Tabular features  : {tabular_features.shape[1]}")
    print(f"  Image features    : {image_features.shape[1]}")
    print(f"  Total features    : {tabular_features.shape[1] + image_features.shape[1]}")
    print(f"  Target (log price): mean={targets.mean():.3f}  std={targets.std():.3f}")
    print(f"  Raw price range   : "
          f"${int(np.expm1(targets.min())):,} – ${int(np.expm1(targets.max())):,}")
    print("=" * 50)
