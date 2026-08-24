# Vegetation Classifier

Binary classifier for **emergent** vs **submergent** aquatic vegetation,
trained on ArcGIS-exported GeoTIFF annotation tiles.

## Project layout

```
veg_classifier/
├── config.yaml          ← all settings live here
├── train.py             ← training script
├── predict.py           ← inference on new tiles
├── requirements.txt
└── utils/
    ├── dataset.py       ← GeoTIFF dataset loader
    ├── model.py         ← ResNet / EfficientNet backbone builder
    └── gradcam.py       ← gradient-weighted visualisations
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare your data

Point `config.yaml → data.root_dir` at your folder of `.tif` tiles.

Your tiles follow the pattern where **every tile contains a single class**
(`num_unique == 1`), so the default `label_strategy: pixel_majority`
will automatically read the dominant pixel value and map it to a class:

```yaml
class_map:
  1: "emergent"
  2: "submergent"
```

## 2. Configure

Edit `config.yaml` to match your setup:

| Key | Default | Notes |
|---|---|---|
| `data.root_dir` | `data/tiles` | path to your .tif files |
| `data.bands` | `[1, 2, 3]` | band indices to use (1-indexed). Use `[1]` for single-band |
| `data.image_size` | 224 | resize all tiles to this square size |
| `model.backbone` | `resnet50` | `resnet18 \| resnet50 \| efficientnet_b0` |
| `training.epochs` | 30 | increase for more tiles |
| `training.batch_size` | 32 | reduce to 8 if GPU memory limited |

## 3. Train

```bash
python train.py
# or with a custom config / data path
python train.py --config config.yaml --data_dir /path/to/tiles
```

Training produces:
- `outputs/checkpoints/best_model.pt` – best validation checkpoint
- `outputs/results/results.json` – accuracy, F1, confusion matrix, loss history
- `outputs/results/gradcam/` – GradCAM overlays for 20 test tiles (if enabled)

## 4. Predict on new tiles

```bash
# Single tile
python predict.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --input data/new_tiles/000000000999.tif

# Whole directory → CSV
python predict.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --input data/new_tiles/ \
  --output predictions.csv
```

## Tips for small datasets

- If you have **< 200 tiles per class**, add more augmentations (flip, rotate, jitter are already on).
- Use `resnet18` instead of `resnet50` to reduce overfitting risk.
- Set `freeze_backbone_epochs: 5` so the head trains first, then fine-tune.
- Class imbalance is handled automatically via inverse-frequency `CrossEntropyLoss` weights.

## Extending to more classes

Add entries to `class_map` in `config.yaml`; the rest of the code adapts automatically:

```yaml
class_map:
  1: "emergent"
  2: "submergent"
  3: "algae"
```

## Multispectral / single-band rasters

Set `data.bands` to whichever bands you want to use:

```yaml
bands: [1]          # single-band annotation mask
bands: [1, 2, 3, 4] # 4-band multispectral
```

The model's first conv layer is automatically resized to match.
