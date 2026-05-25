"""
Evaluation utilities:
  - MAE and RMSE on log-scale and original price scale
  - Actual vs. Predicted scatter plot
  - Training / Validation loss curve plot
  - Full metrics report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Metrics 

def compute_mae(y_true: np.ndarray,
                y_pred: np.ndarray,
                log_scale: bool = True) -> float:
    
    if not log_scale:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 log_scale: bool = True) -> float:
    
    if not log_scale:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def compute_r2(y_true: np.ndarray,
               y_pred: np.ndarray,
               log_scale: bool = True) -> float:
    
    from sklearn.metrics import r2_score
    if not log_scale:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    return float(r2_score(y_true, y_pred))


def print_metrics(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  split_name: str = "Test") -> dict:
    
    mae_log  = compute_mae(y_true,  y_pred, log_scale=True)
    rmse_log = compute_rmse(y_true, y_pred, log_scale=True)
    mae_usd  = compute_mae(y_true,  y_pred, log_scale=False)
    rmse_usd = compute_rmse(y_true, y_pred, log_scale=False)
    r2       = compute_r2(y_true,   y_pred, log_scale=False)

    print(f"\n{'='*48}")
    print(f"  {split_name} Set Metrics")
    print(f"{'='*48}")
    print(f"  MAE  (log scale)  : {mae_log:.4f}")
    print(f"  RMSE (log scale)  : {rmse_log:.4f}")
    print(f"  MAE  (USD)        : ${mae_usd:,.0f}")
    print(f"  RMSE (USD)        : ${rmse_usd:,.0f}")
    print(f"  R²                : {r2:.4f}")
    print(f"{'='*48}\n")

    return {
        "mae_log":  mae_log,
        "rmse_log": rmse_log,
        "mae_usd":  mae_usd,
        "rmse_usd": rmse_usd,
        "r2":       r2,
    }


# ── Plots 

def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     title:  str  = "Actual vs Predicted House Prices",
                     save_path: str = None) -> None:
  
    y_true_usd = np.expm1(y_true)
    y_pred_usd = np.expm1(y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Scatter
    ax.scatter(y_true_usd, y_pred_usd,
               alpha=0.45, s=18, color="#2980b9", label="Predictions")

    # Perfect prediction line
    lim_min = min(y_true_usd.min(), y_pred_usd.min()) * 0.95
    lim_max = max(y_true_usd.max(), y_pred_usd.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "r--", linewidth=1.5, label="Perfect prediction")

    # Trend line
    z = np.polyfit(y_true_usd, y_pred_usd, 1)
    p = np.poly1d(z)
    xs = np.linspace(lim_min, lim_max, 200)
    ax.plot(xs, p(xs), color="#27ae60", linewidth=1.5, label="Trend line")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Actual Sale Price (USD)",    fontsize=12)
    ax.set_ylabel("Predicted Sale Price (USD)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))

    # Annotate R²
    r2  = compute_r2(y_true, y_pred, log_scale=False)
    mae = compute_mae(y_true, y_pred, log_scale=False)
    ax.text(0.05, 0.92, f"R² = {r2:.3f}\nMAE = ${mae:,.0f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_predictions] Saved → '{save_path}'")
    plt.show()


def plot_loss_curves(train_losses: list,
                     val_losses:   list,
                     title:        str  = "Training & Validation Loss (MSE)",
                     save_path:    str  = None) -> None:
    """Line plot of training and validation loss over epochs."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "b-o", markersize=4, linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_losses,   "r-o", markersize=4, linewidth=1.5, label="Val Loss")

    # Mark best val epoch
    best_ep  = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(x=best_ep, color="grey", linestyle="--", linewidth=1,
               label=f"Best val epoch ({best_ep})")
    ax.scatter([best_ep], [best_val], color="red", zorder=5, s=60)

    ax.set_xlabel("Epoch",       fontsize=12)
    ax.set_ylabel("MSE Loss",    fontsize=12)
    ax.set_title(title,          fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_loss_curves] Saved → '{save_path}'")
    plt.show()


def plot_residuals(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   save_path: str = None) -> None:
   
    y_true_usd    = np.expm1(y_true)
    y_pred_usd    = np.expm1(y_pred)
    residuals     = y_pred_usd - y_true_usd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true_usd, residuals,
               alpha=0.4, s=16, color="#8e44ad")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Actual Sale Price (USD)", fontsize=12)
    ax.set_ylabel("Residual  (Pred − Actual, USD)", fontsize=12)
    ax.set_title("Residual Plot", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_residuals] Saved → '{save_path}'")
    plt.show()
