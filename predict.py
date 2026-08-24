"""
predict.py – Run a trained model on new GeoTIFF tiles.

Usage:
    # Single file
    python predict.py --checkpoint outputs/checkpoints/best_model.pt \
                      --config config.yaml \
                      --input data/new_tiles/000000000999.tif

    # Directory of .tif files → writes predictions.csv
    python predict.py --checkpoint outputs/checkpoints/best_model.pt \
                      --config config.yaml \
                      --input data/new_tiles/ \
                      --output predictions.csv
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from utils.dataset import VegetationDataset
from utils.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True, help="File or directory of .tif tiles")
    parser.add_argument("--output", default=None, help="Path for predictions CSV (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = {int(k): v for k, v in cfg["data"]["class_map"].items()}
    num_classes = len(class_map)
    in_channels = len(cfg["data"]["bands"])
    backbone = cfg["model"]["backbone"]

    model = build_model(backbone=backbone, num_classes=num_classes,
                        pretrained=False, in_channels=in_channels,
                        dropout=0.0)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    input_path = Path(args.input)
    if input_path.is_file():
        tif_files = [input_path]
    else:
        tif_files = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff"))

    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_path}")

    from torchvision import transforms
    from utils.dataset import _read_tile_rasterio, _read_tile_tifffile
    import numpy as np

    try:
        import rasterio as _
        use_rasterio = True
    except ImportError:
        use_rasterio = False

    size = cfg["data"]["image_size"]
    bands = cfg["data"]["bands"]
    pct = cfg["data"]["normalize_percentile"]
    in_ch = len(bands)

    if in_ch == 3:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize(mean=[0.5] * in_ch, std=[0.25] * in_ch)

    idx_to_name = {i: v for i, v in enumerate(sorted(class_map.values()))}
    records = []

    print(f"Predicting {len(tif_files)} tile(s)…")
    for fpath in tif_files:
        if use_rasterio:
            arr = _read_tile_rasterio(str(fpath), bands, size, pct)
        else:
            arr = _read_tile_tifffile(str(fpath), bands, size, pct)

        import torch as th
        tensor = th.from_numpy(arr)
        tensor = norm(tensor).unsqueeze(0).to(device)

        with th.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
            pred_idx = logits.argmax(dim=1).item()

        pred_name = idx_to_name[pred_idx]
        conf = probs[pred_idx]
        print(f"  {fpath.name:40s}  →  {pred_name}  (conf={conf:.3f})")
        records.append(dict(
            filename=fpath.name,
            predicted_class=pred_name,
            confidence=round(conf, 4),
            **{f"prob_{idx_to_name[i]}": round(p, 4) for i, p in enumerate(probs)},
        ))

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
