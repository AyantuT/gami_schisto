import torch
from torch.utils.data import DataLoader

from utils.model import UNet
from preprocessing.read_dataset import VegetationDataset


# Dataset
train_dataset = VegetationDataset(
    "data/train/images",
    "data/train/masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet(
    in_channels=3,
    num_classes=3
).to(device)


# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# Training
num_epochs = 20

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, masks)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )