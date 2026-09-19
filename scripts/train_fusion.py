import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, "/home/pavitra/satquery/scripts")
from fusion_model import FusionModel, CLASS_NAMES
from fusion_dataset import BenGeDataset, fusion_collate_fn


def train(
    epochs=15,
    batch_size=64,
    lr=1e-3,
    save_path="/home/pavitra/satquery/checkpoints/fusion_model.pt",
    max_samples=None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Optical-SAR Fusion model on device: {device}")

    # 1. Dataset & Split
    full_dataset = BenGeDataset(max_samples=max_samples)
    total_len = len(full_dataset)
    print(f"Total available dataset samples: {total_len}")
    
    if total_len == 0:
        raise RuntimeError("No data found in BenGeDataset. Ensure data is unpacked.")

    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_ds, val_ds = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train split: {len(train_ds)} samples | Val split: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=fusion_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=fusion_collate_fn
    )

    # 2. Model, Loss, Optimizer
    num_classes = len(CLASS_NAMES)
    model = FusionModel(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    epoch_history = []

    print("\n" + "=" * 65)
    print(f"{'Epoch':8s} | {'Train Loss':12s} | {'Val Loss':12s} | {'LR':10s} | {'Status'}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            sar = batch["sar"].to(device)
            optical = batch["optical"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(sar, optical)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_train_loss / max(train_steps, 1)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                sar = batch["sar"].to(device)
                optical = batch["optical"].to(device)
                target = batch["target"].to(device)

                logits = model(sar, optical)
                loss = criterion(logits, target)
                running_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = running_val_loss / max(val_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            status = "✓ Saved Best"

        epoch_history.append((epoch, avg_train_loss, avg_val_loss))
        print(f"{epoch:6d}/{epochs} | {avg_train_loss:12.4f} | {avg_val_loss:12.4f} | {current_lr:10.2e} | {status}")

    print("=" * 65)
    print(f"Training completed! Best checkpoint saved to: {save_path}")
    print(f"Final Best Validation Loss: {best_val_loss:.4f}\n")
    return epoch_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="/home/pavitra/satquery/checkpoints/fusion_model.pt")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        max_samples=args.max_samples,
    )
