import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class VegetationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".tif")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            image_name
        )

        # Read image & mask using rasterio
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Convert image:
        # (bands, height, width)
        #  the format PyTorch wants

        # Normalization
        image = image / 255.0

        # Convert mask labels to model classes
        # Adjust these depending on actual labels
        new_mask = np.zeros_like(mask, dtype=np.int64)

        new_mask[mask == 1] = 1      # emergent
        new_mask[mask == 3] = 2      # loamy soil

        # If submergent is represented by another value:
        # new_mask[mask == X] = 2

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(new_mask, dtype=torch.long)

        return image, mask