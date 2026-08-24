"""
train.py – Fine-tune a CNN backbone for vegetation type classification.

Usage:
    python train.py                        # uses config.yaml in current dir
    python train.py --config my_cfg.yaml   # custom config path
    python train.py --config config.yaml --data_dir /path/to/tiles
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import yaml


# ── local imports ─────────────────────────────────────────────────────────────
from utils.dataset import VegetationDataset
from utils.model import build_model, freeze_backbone, unfreeze_all, get_gradcam_layer
from utils.gradcam import GradCAM, save_gradcam_figure


# ── helpers ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(cfg, train: bool):
    """Build torchvision transforms for train or val/test."""
    aug = cfg["augmentation"]
    size = cfg["data"]["image_size"]

    ops = [transforms.Resize((size, size))]

    if train:
        scale = aug.get("random_crop_scale", [0.8, 1.0])
        ops.append(transforms.RandomResizedCrop(size, scale=tuple(scale)))
        if aug.get("horizontal_flip"):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip"):
            ops.append(transforms.RandomVerticalFlip())
        degrees = aug.get("rotate_degrees", 0)
        if degrees:
            ops.append(transforms.RandomRotation(degrees))
        jitter = aug.get("color_jitter", 0)
        if jitter:
            ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter,
                                              saturation=jitter, hue=jitter / 2))

    # Normalise per-band; ImageNet stats are a reasonable default for 3-band
    in_ch = len(cfg["data"]["bands"])
    if in_ch == 3:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
    else:
        ops.append(transforms.Normalize(mean=[0.5] * in_ch, std=[0.25] * in_ch))

    return transforms.Compose(ops)


def make_splits(dataset: VegetationDataset, cfg: dict):
    """Stratified train/val/test split returning index lists."""
    split = cfg["split"]
    seed = split["seed"]
    labels = dataset.labels
    indices = list(range(len(dataset)))

    test_frac = split["test"]
    val_frac = split["val"] / (1 - test_frac)  # val fraction of remaining

    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_frac, random_state=seed,
        stratify=labels if split.get("stratify") else None,
    )
    lbls_trainval = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, random_state=seed,
        stratify=lbls_trainval if split.get("stratify") else None,
    )
    return idx_train, idx_val, idx_test


def evaluate(model, loader, criterion, device, amp=False):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=amp):
                logits = model(imgs)
                loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            n += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / n, correct / n, all_preds, all_labels


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default=None, help="Override data.root_dir in config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["root_dir"] = args.data_dir

    seed_everything(cfg["split"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
    print(f"Device: {device}  |  AMP: {amp}")

    # ── Directories ───────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    res_dir = Path(cfg["output"]["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    class_map = {int(k): v for k, v in cfg["data"]["class_map"].items()}
    full_ds = VegetationDataset(
        root_dir=cfg["data"]["root_dir"],
        class_map=class_map,
        label_strategy=cfg["data"]["label_strategy"],
        label_regex=cfg["data"].get("label_regex"),
        bands=cfg["data"]["bands"],
        image_size=cfg["data"]["image_size"],
        percentile_clip=cfg["data"]["normalize_percentile"],
    )
    full_ds.summary()

    idx_train, idx_val, idx_test = make_splits(full_ds, cfg)
    print(f"Split  train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    train_transform = get_transforms(cfg, train=True)
    val_transform = get_transforms(cfg, train=False)

    # Apply correct transforms per split
    # (We reuse full_ds index lists with per-split datasets carrying their own transform)
    train_ds = VegetationDataset(
        root_dir=cfg["data"]["root_dir"],
        class_map=class_map,
        label_strategy=cfg["data"]["label_strategy"],
        label_regex=cfg["data"].get("label_regex"),
        bands=cfg["data"]["bands"],
        image_size=cfg["data"]["image_size"],
        percentile_clip=cfg["data"]["normalize_percentile"],
        transform=train_transform,
        file_list=[full_ds.files[i].name for i in idx_train],
    )
    val_ds = VegetationDataset(
        root_dir=cfg["data"]["root_dir"],
        class_map=class_map,
        label_strategy=cfg["data"]["label_strategy"],
        label_regex=cfg["data"].get("label_regex"),
        bands=cfg["data"]["bands"],
        image_size=cfg["data"]["image_size"],
        percentile_clip=cfg["data"]["normalize_percentile"],
        transform=val_transform,
        file_list=[full_ds.files[i].name for i in idx_val],
    )
    test_ds = VegetationDataset(
        root_dir=cfg["data"]["root_dir"],
        class_map=class_map,
        label_strategy=cfg["data"]["label_strategy"],
        label_regex=cfg["data"].get("label_regex"),
        bands=cfg["data"]["bands"],
        image_size=cfg["data"]["image_size"],
        percentile_clip=cfg["data"]["normalize_percentile"],
        transform=val_transform,
        file_list=[full_ds.files[i].name for i in idx_test],
    )

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone = cfg["model"]["backbone"]
    in_channels = len(cfg["data"]["bands"])
    num_classes = len(class_map)

    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=cfg["model"]["pretrained"],
        in_channels=in_channels,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Class weights for imbalanced data
    class_weights = full_ds.class_weights().to(device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        freeze_backbone(model, backbone)
        print(f"Backbone frozen for first {freeze_epochs} epochs.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    sched_type = cfg["training"].get("scheduler", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    scaler = GradScaler(enabled=amp)
    patience = cfg["training"].get("early_stopping_patience", 10)
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            unfreeze_all(model)
            print(f"Epoch {epoch}: backbone unfrozen, rebuilding optimizer.")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"] * 0.1,  # lower LR for fine-tuning
                weight_decay=cfg["training"]["weight_decay"],
            )

        model.train()
        epoch_loss, correct, n = 0.0, 0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=amp):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            preds = logits.detach().argmax(dim=1)
            epoch_loss += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            n += len(labels)

        train_loss = epoch_loss / n
        train_acc = correct / n
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, amp)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        row = dict(epoch=epoch, train_loss=round(train_loss, 4),
                   train_acc=round(train_acc, 4), val_loss=round(val_loss, 4),
                   val_acc=round(val_acc, 4))
        history.append(row)
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n── Test evaluation ──")
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
    test_loss, test_acc, preds, true_labels = evaluate(model, test_loader, criterion, device, amp)
    print(f"Test loss: {test_loss:.4f}   Test accuracy: {test_acc:.4f}")

    # Confusion matrix & per-class F1
    from sklearn.metrics import classification_report, confusion_matrix
    idx_to_name = full_ds.idx_to_name
    target_names = [idx_to_name[i] for i in sorted(idx_to_name)]
    report = classification_report(true_labels, preds, target_names=target_names)
    cm = confusion_matrix(true_labels, preds)
    print(report)
    print("Confusion matrix:\n", cm)

    results = dict(
        test_loss=round(test_loss, 4),
        test_acc=round(test_acc, 4),
        classification_report=report,
        confusion_matrix=cm.tolist(),
        history=history,
    )
    with open(res_dir / "results.json", "w") as f:
        # report is str so keep it there
        r = {k: (v if not isinstance(v, str) else v) for k, v in results.items()}
        json.dump(r, f, indent=2)
    print(f"Results saved to {res_dir / 'results.json'}")

    # ── GradCAM ───────────────────────────────────────────────────────────────
    if cfg["output"].get("gradcam"):
        print("\n── Generating GradCAM visualizations ──")
        gradcam_dir = res_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)
        target_layer = get_gradcam_layer(model, backbone)
        gcam = GradCAM(model, target_layer)
        model.eval()

        # Generate for up to 20 test tiles
        n_samples = min(20, len(test_ds))
        for i in range(n_samples):
            img, lbl = test_ds[i]
            inp = img.unsqueeze(0).to(device)
            cam = gcam.compute(inp)
            pred = model(inp).argmax(dim=1).item()
            save_gradcam_figure(
                tile=img,
                cam=cam,
                true_label=idx_to_name[lbl],
                pred_label=idx_to_name[pred],
                save_path=str(gradcam_dir / f"tile_{i:04d}.png"),
            )
        gcam.remove_hooks()
        print(f"GradCAM images saved to {gradcam_dir}")

    print("\nDone! Best val accuracy:", round(best_val_acc, 4))


if __name__ == "__main__":
    main()
