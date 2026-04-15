"""
train_attention_model.py
------------------------
Trains the CTA TransitDelayPredictor on delay_records.csv.

Uses Huber loss (smooth L1) which is robust to the outlier delays that appear
during blizzards and post-game surges. Saves the best checkpoint (lowest
validation loss) to models/transit_attention.pt.

After training, the full pipeline becomes live:
  1. python scripts/generate_llama_data.py --model models/transit_attention.pt
  2. python src/train_llama.py
  3. streamlit run app.py

Run:
    python scripts/train_attention_model.py
    python scripts/train_attention_model.py --epochs 100 --lr 3e-4 --data data/delay_records.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.attention_model import TransitDelayPredictor, ModelConfig, WeatherBias

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DelayDataset(Dataset):
    """
    Wraps delay_records.csv.

    Target is the *residual* after subtracting the WeatherBias prior:
        residual = delay_minutes - weather_prior(weather_idx)

    This means the attention model only needs to learn station crowding,
    sports surges, and time-of-day effects.  Weather is handled by the
    fixed research-grounded prior so the model doesn't try to re-learn
    something that ridership data cannot reliably teach it.

    The residual is normalised by residual_mean (its own mean, not the
    raw delay mean) so the model sees balanced magnitudes.  De-normalise
    at inference by multiplying back by residual_mean (stored as delay_mean
    in the checkpoint for backwards compatibility).
    """

    def __init__(self, df: pd.DataFrame, delay_mean: float | None = None):
        self.station_ids = torch.tensor(df["station_id"].values, dtype=torch.long)
        self.time_of_day = torch.tensor(df["time_norm"].values,  dtype=torch.float)
        self.weather_idx = torch.tensor(df["weather_idx"].values, dtype=torch.float)
        self.sports_flag = torch.tensor(df["sports_flag"].values, dtype=torch.float)

        # Compute weather prior for every row using the same fixed table used at inference
        _prior_fn = WeatherBias()
        w_tensor  = torch.tensor(df["weather_idx"].values, dtype=torch.float)
        with torch.no_grad():
            weather_prior_vals = _prior_fn(w_tensor).numpy()

        raw_delays = df["delay_minutes"].values.astype(np.float32)
        # Residual = total observed delay minus the research-grounded weather component
        residuals  = np.clip(raw_delays - weather_prior_vals, 0.0, None)

        self.delay_mean = float(delay_mean or residuals.mean())
        if self.delay_mean == 0.0:
            self.delay_mean = 1.0  # safeguard against degenerate data
        self.delays = torch.tensor(residuals / self.delay_mean, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.delays)

    def __getitem__(self, idx: int) -> dict:
        return {
            "station_ids":   self.station_ids[idx],
            "time_of_day":   self.time_of_day[idx],
            "weather_index": self.weather_idx[idx],
            "sports_event":  self.sports_flag[idx],
            "target":        self.delays[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, delay_mean: float) -> dict:
    """Returns MAE and RMSE in real minutes."""
    p = preds.detach().cpu() * delay_mean
    t = targets.detach().cpu() * delay_mean
    mae  = (p - t).abs().mean().item()
    rmse = ((p - t) ** 2).mean().sqrt().item()
    return {"mae": round(mae, 3), "rmse": round(rmse, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict) -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = Path(cfg["data"])
    if not data_path.exists():
        print(f"[train_attn] ERROR: {data_path} not found.")
        print("             Run:  python scripts/build_training_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"[train_attn] Loaded {len(df):,} rows from {data_path}")

    # Compute residual mean on the FULL dataset before splitting so val/test
    # are normalised consistently.  The residual = delay_minutes - weather_prior,
    # so this mean is smaller than raw delay_mean — that's expected.
    _prior_fn = WeatherBias()
    w_all = torch.tensor(df["weather_idx"].values, dtype=torch.float)
    with torch.no_grad():
        prior_all = _prior_fn(w_all).numpy()
    residuals_all = np.clip(df["delay_minutes"].values.astype(np.float32) - prior_all, 0.0, None)
    delay_mean = float(residuals_all.mean()) or 1.0
    print(f"[train_attn] Raw delay mean   : {df['delay_minutes'].mean():.2f} min")
    print(f"[train_attn] Weather prior mean: {prior_all.mean():.2f} min  (from WeatherBias research table)")
    print(f"[train_attn] Residual mean     : {delay_mean:.2f} min  (target for attention model)")

    full_dataset = DelayDataset(df, delay_mean=delay_mean)

    # 80 / 10 / 10 split
    n         = len(full_dataset)
    n_train   = int(0.80 * n)
    n_val     = int(0.10 * n)
    n_test    = n - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    print(f"[train_attn] Split — train: {n_train:,}  val: {n_val:,}  test: {n_test:,}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_attn] Device: {device}")

    model_cfg = ModelConfig(
        num_stations = 150,
        embed_dim    = cfg["embed_dim"],
        num_heads    = cfg["num_heads"],
        dropout      = cfg["dropout"],
    )
    model = TransitDelayPredictor(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_attn] Trainable parameters: {n_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # Cosine decay to 5 % of initial LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.05
    )

    # Huber loss — behaves like MSE near zero, like MAE for large residuals
    criterion = nn.HuberLoss(delta=1.0)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val MAE':>9}  {'Val RMSE':>10}  {'LR':>10}")
    print("─" * 65)

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train pass ──────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            preds = model(
                batch["station_ids"].to(device),
                batch["time_of_day"].to(device),
                batch["weather_index"].to(device),
                batch["sports_event"].to(device),
            )
            # Model output is already normalised via Softplus; target is normalised too
            loss = criterion(preds, batch["target"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss_sum += loss.item() * len(batch["target"])

        train_loss = train_loss_sum / n_train
        scheduler.step()

        # ── val pass ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(
                    batch["station_ids"].to(device),
                    batch["time_of_day"].to(device),
                    batch["weather_index"].to(device),
                    batch["sports_event"].to(device),
                )
                val_loss_sum += criterion(preds, batch["target"].to(device)).item() * len(batch["target"])
                all_preds.append(preds.cpu())
                all_targets.append(batch["target"])

        val_loss    = val_loss_sum / n_val
        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics     = compute_metrics(all_preds, all_targets, delay_mean)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(metrics["mae"])
        history["val_rmse"].append(metrics["rmse"])

        lr_now = optimiser.param_groups[0]["lr"]
        marker = " ✓" if val_loss < best_val_loss else ""
        print(
            f"{epoch:>6}  {train_loss:>11.5f}  {val_loss:>10.5f}  "
            f"{metrics['mae']:>8.2f}m  {metrics['rmse']:>9.2f}m  {lr_now:>10.2e}{marker}"
        )

        # ── checkpoint best model ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save delay_mean alongside weights so inference can de-normalise
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config":     model_cfg,
                    "delay_mean": delay_mean,
                },
                out_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\n[train_attn] Early stop — no improvement for {cfg['patience']} epochs.")
                break

    print(f"\n[train_attn] Best val loss: {best_val_loss:.5f}")
    print(f"[train_attn] Model saved  → {out_path}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            p = model(
                batch["station_ids"].to(device),
                batch["time_of_day"].to(device),
                batch["weather_index"].to(device),
                batch["sports_event"].to(device),
            )
            test_preds.append(p.cpu())
            test_targets.append(batch["target"])

    test_metrics = compute_metrics(torch.cat(test_preds), torch.cat(test_targets), delay_mean)
    print(f"\n── Test set results ─────────────────────────────────────")
    print(f"   MAE  : {test_metrics['mae']:.2f} min")
    print(f"   RMSE : {test_metrics['rmse']:.2f} min")

    # ── Optional: training curve plot ────────────────────────────────────────
    if MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_x = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs_x, history["train_loss"], label="Train loss")
        ax1.plot(epochs_x, history["val_loss"],   label="Val loss")
        ax1.set_title("Huber Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

        ax2.plot(epochs_x, history["val_mae"],  label="Val MAE (min)")
        ax2.plot(epochs_x, history["val_rmse"], label="Val RMSE (min)")
        ax2.set_title("Validation Error (minutes)"); ax2.set_xlabel("Epoch"); ax2.legend()

        plot_path = out_path.parent / "training_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        print(f"\n[train_attn] Training curve → {plot_path}")

    # ── Attention weight sanity check ─────────────────────────────────────────
    print("\n── Attention weight sanity check ────────────────────────")
    print("   (Each scenario below is dominated by one feature — model should reflect that)\n")

    checks = [
        dict(label="PM rush, clear, no event  → expect Time_of_Day",
             station_id=20, time_norm=17.0/24, weather_idx=0.02, sports_flag=0.0),
        dict(label="Midday, blizzard, no event → expect Weather_Index",
             station_id=10, time_norm=12.0/24, weather_idx=0.90, sports_flag=0.0),
        dict(label="Evening, clear, Cubs game  → expect Sports_Event",
             station_id=12, time_norm=19.0/24, weather_idx=0.02, sports_flag=1.0),
        dict(label="Late night, clear, no event → expect Time_of_Day",
             station_id=5,  time_norm= 2.0/24, weather_idx=0.05, sports_flag=0.0),
    ]

    model.eval()
    for chk in checks:
        result = model.explain(
            torch.tensor([chk["station_id"]],  dtype=torch.long),
            torch.tensor([chk["time_norm"]],   dtype=torch.float),
            torch.tensor([chk["weather_idx"]], dtype=torch.float),
            torch.tensor([chk["sports_flag"]], dtype=torch.float),
        )
        # De-normalise the delay
        raw_delay = result["predicted_delay_minutes"] * delay_mean
        imp = {k: f"{v:.0%}" for k, v in result["feature_importance"].items()}
        print(f"   {chk['label']}")
        print(f"     delay={raw_delay:.1f} min  dominant={result['dominant_factor']}  {imp}\n")

    print("── Next steps ────────────────────────────────────────────")
    print(f"   python scripts/generate_llama_data.py --model {out_path}")
    print(f"   python src/train_llama.py")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTA TransitDelayPredictor")
    parser.add_argument("--data",      default=str(ROOT / "data" / "delay_records.csv"))
    parser.add_argument("--output",    default=str(ROOT / "models" / "transit_attention.pt"))
    parser.add_argument("--epochs",    type=int,   default=80)
    parser.add_argument("--batch",     type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--patience",  type=int,   default=15,
                        help="Early-stop after N epochs with no val improvement")
    parser.add_argument("--embed-dim", type=int,   default=32,  dest="embed_dim")
    parser.add_argument("--num-heads", type=int,   default=4,   dest="num_heads")
    parser.add_argument("--dropout",   type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    train(vars(args))
