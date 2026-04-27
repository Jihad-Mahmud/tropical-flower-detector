# train_classifier.py  (Windows-fixed version)
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Train a small classifier head on top of the frozen BYOL backbone.
# RUN ONCE: python train_classifier.py
# OUTPUT  : models/flower_classifier.pth   ← used by the web server
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import FlowerClassifier, load_byol_backbone_weights, CLASS_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
BYOL_WEIGHTS = "models/byol_backbone.pth"
SAVE_PATH    = "models/flower_classifier.pth"

EPOCHS       = 25
BATCH_SIZE   = 32
LR           = 1e-3
VAL_SPLIT    = 0.15
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Windows fix: num_workers=0 prevents the multiprocessing RuntimeError
NUM_WORKERS  = 0
PIN_MEMORY   = False


# IMPORTANT: On Windows, all training code MUST be inside if __name__ == '__main__'
# This is what caused the RuntimeError you saw before.
if __name__ == '__main__':

    print("=" * 60)
    print("  Flower Classifier — Training Script")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Dataset dir : {DATASET_DIR}")
    print(f"  BYOL weights: {BYOL_WEIGHTS}")
    print(f"  Save path   : {SAVE_PATH}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print("=" * 60)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset folder not found: '{DATASET_DIR}'\n"
            f"Please create:  flower-detector/dataset/\n"
            f"with sub-folders: {CLASS_NAMES}"
        )

    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
    print(f"\nDataset classes found : {full_dataset.classes}")
    print(f"Total images          : {len(full_dataset)}")

    # Train / Val split
    n_val   = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_tf

    # num_workers=0 and pin_memory=False — required for Windows
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    print(f"Train samples : {n_train}  |  Val samples : {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FlowerClassifier().to(DEVICE)
    load_byol_backbone_weights(model, BYOL_WEIGHTS, DEVICE)

    # Freeze backbone — only train the small classifier head
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters  : {total/1e6:.1f}M total  |  {trainable/1e3:.1f}K trainable (head only)")

    # ── Training ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        t_loss = t_correct = t_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            _, preds   = outputs.max(1)
            t_correct += preds.eq(labels).sum().item()
            t_total   += labels.size(0)

        # Validate
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs   = model(imgs)
                loss      = criterion(outputs, labels)
                v_loss   += loss.item()
                _, preds   = outputs.max(1)
                v_correct += preds.eq(labels).sum().item()
                v_total   += labels.size(0)

        train_acc = 100. * t_correct / t_total
        val_acc   = 100. * v_correct / v_total
        scheduler.step()

        is_best = val_acc > best_val_acc
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train Loss: {t_loss/len(train_loader):.4f}  "
              f"Train Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {v_loss/len(val_loader):.4f}  "
              f"Val Acc: {val_acc:.2f}%"
              + ("  <- best" if is_best else ""))

        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nTraining complete!  Best val accuracy: {best_val_acc:.2f}%")
    print(f"Saved to: {SAVE_PATH}")
    print(f"\nNext step: run the web server with:")
    print(f"  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
