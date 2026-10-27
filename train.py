"""Fine-tune DenseNet-121 for NORMAL vs PNEUMONIA.

    python train.py --data data/raw/chest_xray --epochs 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from xray.data import build_loaders
from xray.metrics import clinical_metrics, threshold_for_sensitivity
from xray.model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).squeeze(1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, test_dl, classes = build_loaders(args.data, batch_size=args.batch_size)
    assert classes == ["NORMAL", "PNEUMONIA"], classes

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(x)

        y_val, p_val = predict(model, val_dl, device)
        auc = roc_auc_score(y_val, p_val)
        print(f"epoch {epoch:2d}  loss {running / len(train_dl.dataset):.4f}  val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    y_val, p_val = predict(model, val_dl, device)
    threshold = threshold_for_sensitivity(y_val, p_val, args.target_sensitivity)
    y_test, p_test = predict(model, test_dl, device)
    report = {
        "val_auc": float(roc_auc_score(y_val, p_val)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_at_0.5": clinical_metrics(y_test, p_test, 0.5),
        "test_at_tuned_threshold": clinical_metrics(y_test, p_test, threshold),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "threshold.json").write_text(json.dumps({"threshold": threshold}))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
