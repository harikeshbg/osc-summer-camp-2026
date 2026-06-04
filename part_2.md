# Plant Disease Detection with AI — Part 2 Code

## Part 2: Train a Plant Disease Classifier with Transfer Learning

In this part, we use a pretrained ResNet18 model and fine-tune it for plant disease classification.

The goal is to:

* Load the plant disease dataset.
* Prepare images for the model.
* Load a pretrained ResNet18 model.
* Replace its final layer for plant disease classes.
* Train the model.
* Save the best model.
* Plot training results.

---

## Cell 1 — Imports and GPU Check

```python
import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# Makes results more repeatable.
torch.manual_seed(42)

# Check PyTorch version.
print("PyTorch version:", torch.__version__)

# Check whether GPU is available.
print("CUDA available:", torch.cuda.is_available())

# Use GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Print GPU name if available.
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Cell 2 — Dataset Path

```python
# Main dataset folder.
DATASET_PATH = "/fs/ess/PAS2699/osc_summer_camp_2026/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

# Training images are used for learning.
TRAIN_DIR = os.path.join(DATASET_PATH, "train")

# Validation images are used for checking model performance.
VALID_DIR = os.path.join(DATASET_PATH, "valid")

# Check whether the folders exist.
print("Train exists:", os.path.exists(TRAIN_DIR))
print("Valid exists:", os.path.exists(VALID_DIR))

# Count how many class folders are present.
print("Train classes:", len(os.listdir(TRAIN_DIR)))
print("Valid classes:", len(os.listdir(VALID_DIR)))
```

Expected output:

```text
Train exists: True
Valid exists: True
Train classes: 38
Valid classes: 38
```

---

## Cell 3 — Image Transforms and Datasets

```python
# ResNet18 expects images around 224 x 224.
IMG_SIZE = 224

# Number of images processed together at once.
BATCH_SIZE = 32

# Transform for training images.
train_transform = transforms.Compose([
    # Resize every image to the same size.
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Convert image to PyTorch tensor format.
    transforms.ToTensor(),

    # Normalize using ImageNet values because ResNet18 was trained this way.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# Transform for validation images.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ImageFolder reads images from folders.
# Each folder name becomes a class label.
train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
valid_dataset = ImageFolder(root=VALID_DIR, transform=val_transform)

# Number of plant disease classes.
NUM_CLASSES = len(train_dataset.classes)

print("Number of classes:", NUM_CLASSES)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(valid_dataset))

# Print class index and class name.
print("\nClasses:")
for i, cls in enumerate(train_dataset.classes):
    print(f"{i:>2}: {cls}")
```

---

## Cell 4 — DataLoaders

```python
# DataLoader sends images to the model in batches.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # Shuffle training images so the model sees mixed examples.
    num_workers=2,      # Use background workers to load data.
    pin_memory=True     # Helps speed up GPU transfer.
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,      # No need to shuffle validation images.
    num_workers=2,
    pin_memory=True
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(valid_loader))
```

---

## Cell 5 — Load ResNet18 and Replace Final Layer

```python
print("Loading pretrained ResNet18...")

# Load ResNet18 with pretrained ImageNet weights.
# This is the transfer learning step.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Show original final layer.
print("Original final layer:")
print(model.fc)

# Get number of input features going into the final layer.
in_features = model.fc.in_features

# Replace the final layer.
# Original model predicts 1000 ImageNet classes.
# New model predicts NUM_CLASSES plant disease classes.
model.fc = nn.Linear(in_features, NUM_CLASSES)

# Show new final layer.
print("\nNew final layer:")
print(model.fc)

# Move model to GPU or CPU.
model = model.to(device)

# Count total and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel ready.")
print("Total parameters:", f"{total_params:,}")
print("Trainable parameters:", f"{trainable_params:,}")
```

If pretrained weight download fails, use this only to test whether the pipeline runs:

```python
model = models.resnet18(weights=None)
```

For real transfer learning, use:

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

---

## Cell 6 — Loss Function, Optimizer, and Scheduler

```python
# Loss function measures how wrong the model is.
# CrossEntropyLoss is used for multi-class classification.
criterion = nn.CrossEntropyLoss()

# Optimizer updates the model weights after mistakes.
# lr means learning rate.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler lowers the learning rate if validation loss stops improving.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)

print("Loss function: CrossEntropyLoss")
print("Optimizer: Adam, lr=0.001")
print("Scheduler: ReduceLROnPlateau")
```

---

## Cell 7 — Training and Evaluation Functions

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one full pass through the training dataset.
    """

    # Put model in training mode.
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    # Loop through batches of images and labels.
    for batch_idx, (images, labels) in enumerate(loader):

        # Move images and labels to GPU or CPU.
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradient values.
        optimizer.zero_grad()

        # Forward pass: model makes predictions.
        outputs = model(images)

        # Compare predictions with correct labels.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients.
        loss.backward()

        # Optimizer updates model weights.
        optimizer.step()

        # Add this batch's loss.
        total_loss += loss.item()

        # Pick the class with the highest score.
        predicted = outputs.argmax(dim=1)

        # Count correct predictions.
        correct += (predicted == labels).sum().item()

        # Count total images processed.
        total += labels.size(0)

        # Print progress every 50 batches.
        if (batch_idx + 1) % 50 == 0:
            running_acc = 100 * correct / total
            print(
                f"Batch {batch_idx + 1}/{len(loader)} "
                f"| Loss: {loss.item():.4f} "
                f"| Running Acc: {running_acc:.2f}%"
            )

    # Average loss for this epoch.
    avg_loss = total_loss / len(loader)

    # Accuracy for this epoch.
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation data.
    The model does not learn during evaluation.
    """

    # Put model in evaluation mode.
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # Disable gradient calculation during evaluation.
    with torch.no_grad():

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Model makes predictions.
            outputs = model(images)

            # Calculate validation loss.
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Pick highest scoring class.
            predicted = outputs.argmax(dim=1)

            # Count correct predictions.
            correct += (predicted == labels).sum().item()

            # Count total images.
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
```

---

## Cell 8 — Train the Model

```python
# Start with 1 epoch for testing.
# Increase to 5 later for a full run.
NUM_EPOCHS = 1

# Lists to store training history.
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Track best validation accuracy.
best_val_acc = 0.0

print(f"Starting training for {NUM_EPOCHS} epoch(s)")
print("=" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-" * 50)

    # Train for one epoch.
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # Evaluate on validation data.
    val_loss, val_acc = evaluate(
        model,
        valid_loader,
        criterion,
        device
    )

    # Update learning rate scheduler using validation loss.
    scheduler.step(val_loss)

    # Save history for plotting.
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"\nTrain | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

    # Save model if it has the best validation accuracy so far.
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "resnet18_plantdisease_part2_best.pth")
        print(f"New best model saved. Val Acc: {best_val_acc:.2f}%")

    print("=" * 70)

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
```

---

## Cell 9 — Check Saved Model

```python
# Check whether the trained model file was created.
print("Model saved:", os.path.exists("resnet18_plantdisease_part2_best.pth"))
```

Expected output:

```text
Model saved: True
```

---

## Cell 10 — Plot Learning Curves

```python
epochs = range(1, NUM_EPOCHS + 1)

# Plot training and validation loss.
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Part 2 ResNet18 Training — Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("part2_loss_curve.png", dpi=150)
plt.show()

# Plot training and validation accuracy.
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
plt.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Part 2 ResNet18 Training — Accuracy")
plt.ylim(0, 105)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("part2_accuracy_curve.png", dpi=150)
plt.show()

print("Saved plots:")
print("part2_loss_curve.png")
print("part2_accuracy_curve.png")
```

---

## Cell 11 — Check Output Files

```python
files_to_check = [
    "resnet18_plantdisease_part2_best.pth",
    "part2_loss_curve.png",
    "part2_accuracy_curve.png"
]

for f in files_to_check:
    print(f, "exists:", os.path.exists(f))
```

---

## Optional: Run Full Training

After the 1-epoch test works, rerun the training with:

```python
NUM_EPOCHS = 5
```

Then rerun the training and plotting cells.
