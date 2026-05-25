"""
Fusion regression model that combines tabular features + CNN image features.

Architecture:
  tabular (T,) ──┐
                 ├─► Concat (T+1280,) ─► FC(512) ─► ReLU ─► Dropout
  image (1280,) ─┘                    ─► FC(256) ─► ReLU ─► Dropout
                                      ─► FC(1)  (log SalePrice)
"""

import torch
import torch.nn as nn


# ── Fusion Model 

class FusionModel(nn.Module):
    def __init__(self,
                 tabular_dim:  int,
                 image_dim:    int   = 1280,
                 hidden_dims:  list  = None,
                 dropout_p:    float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        input_dim = tabular_dim + image_dim

        # Build the MLP fusion head dynamically
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
            in_dim = h_dim

        # Final regression output (single neuron, no activation)
        layers.append(nn.Linear(in_dim, 1))

        self.fusion_head = nn.Sequential(*layers)

        # Weight initialisation (He init for ReLU networks)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                x_tab: torch.Tensor,
                x_img: torch.Tensor) -> torch.Tensor:
        x   = torch.cat([x_tab, x_img], dim=1)   # (B, tabular_dim + image_dim)
        out = self.fusion_head(x)                 # (B, 1)
        return out.squeeze(1)                     # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training Step 

def train_one_epoch(model:     FusionModel,
                    loader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device) -> float:
    model.train()
    total_loss = 0.0

    for x_tab, x_img, y in loader:
        x_tab = x_tab.to(device)
        x_img = x_img.to(device)
        y     = y.to(device)

        optimizer.zero_grad()
        preds = model(x_tab, x_img)
        loss  = criterion(preds, y)
        loss.backward()

        # Gradient clipping (helps with stability on small datasets)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


# ── Validation Step 

@torch.no_grad()
def evaluate_epoch(model:     FusionModel,
                   loader,
                   criterion: nn.Module,
                   device:    torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_true   = []

    for x_tab, x_img, y in loader:
        x_tab = x_tab.to(device)
        x_img = x_img.to(device)
        y     = y.to(device)

        preds = model(x_tab, x_img)
        loss  = criterion(preds, y)

        total_loss += loss.item() * len(y)
        all_preds.append(preds.cpu())
        all_true.append(y.cpu())

    mean_loss = total_loss / len(loader.dataset)
    y_pred    = torch.cat(all_preds).numpy()
    y_true    = torch.cat(all_true).numpy()

    return mean_loss, y_true, y_pred


# ── Model I/O 

def save_model(model: FusionModel, path: str) -> None:
    """Save model state dict."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[save_model] Model saved → '{path}'")


def load_model(path: str,
               tabular_dim: int,
               image_dim:   int   = 1280,
               hidden_dims: list  = None,
               device: torch.device = None) -> FusionModel:
    if device is None:
        device = torch.device("cpu")
    model = FusionModel(tabular_dim, image_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[load_model] Model loaded from '{path}'")
    return model
