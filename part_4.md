# Plant Disease Detection with AI — Part 4 Instructor Monitor

## Part 4: Data Augmentation Experiment

In Part 4, students compare two experiments:

```text
Experiment A: Train without augmentation
Experiment B: Train with augmentation
```

The goal is to understand whether random image changes can help the model become more robust.

---

# Big Picture Explanation

Use this before starting:

> In real life, leaf photos may be taken from different angles, distances, and lighting conditions. Data augmentation creates random variations of training images so the model practices with more realistic image variation.

Simple flow:

```text
Load saved Part 2 model
        ↓
Create normal training pipeline
        ↓
Create augmented training pipeline
        ↓
Train both experiments
        ↓
Compare validation accuracy and loss
        ↓
Check if augmentation helped
```

---

# Cell 1 — Imports and Device Check

```python
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set seeds so random operations are more repeatable.
# This does not make everything perfectly identical, but helps.
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Use GPU if available. Otherwise, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Print GPU name if CUDA is available.
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Explain After Cell 1

This cell loads all libraries needed for augmentation, training, and plotting.

Important terms:

* `torch`: core deep learning library.
* `torchvision.transforms`: image preprocessing and augmentation tools.
* `ImageFolder`: reads images from folder-based datasets.
* `DataLoader`: loads images in batches.
* `PIL.Image`: opens image files for visualization.

## Ask Students

### Q1. Why do we need GPU?

**Answer:**
Training deep learning models requires many calculations. GPU makes training faster.

### Q2. Are we starting from scratch?

**Answer:**
No. We will start from the saved Part 2 model and continue training.

---

# Cell 2 — Dataset and Model Paths

```python
# Main dataset folder.
DATASET_PATH = "/fs/ess/PAS2699/osc_summer_camp_2026/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

# Training images are used for learning.
TRAIN_DIR = os.path.join(DATASET_PATH, "train")

# Validation images are used for fair evaluation.
VALID_DIR = os.path.join(DATASET_PATH, "valid")

# This is the trained model saved from Part 2.
# Both experiments will start from this same model.
BASE_MODEL_PATH = "resnet18_plantdisease_part2_best.pth"

# Check that all required folders/files exist.
print("Train exists:", os.path.exists(TRAIN_DIR))
print("Valid exists:", os.path.exists(VALID_DIR))
print("Base model exists:", os.path.exists(BASE_MODEL_PATH))
```

Expected output:

```text
Train exists: True
Valid exists: True
Base model exists: True
```

## Explain After Cell 2

We need:

1. Training images.
2. Validation images.
3. Saved Part 2 model.

Both experiments must start from the same saved model to make the comparison fair.

## Ask Students

### Q1. Why do both experiments start from the same model?

**Answer:**
So the only major difference is whether augmentation is used.

### Q2. What happens if `Base model exists` is `False`?

**Answer:**
The notebook cannot load the trained Part 2 model. We need to check the file path or rerun Part 2.

---

# Cell 3 — Basic Settings

```python
# ResNet18 expects images around 224 x 224.
IMG_SIZE = 224

# Number of images processed together at once.
BATCH_SIZE = 32

# Start with 1 epoch to test the pipeline.
# Increase later to 3 or 5 for a full run.
NUM_EPOCHS = 1

print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
```

## Explain After Cell 3

* `IMG_SIZE = 224`: all images are resized to 224 x 224.
* `BATCH_SIZE = 32`: model processes 32 images at a time.
* `NUM_EPOCHS = 1`: model goes through the full training dataset once.

## Ask Students

### Q1. Why start with one epoch?

**Answer:**
To test that the notebook runs correctly before doing longer training.

### Q2. What is a batch?

**Answer:**
A batch is a small group of images processed together.

---

# Cell 4 — Visualize Augmentation

```python
# This transform is only for showing what augmentation looks like.
# It is not the final training transform yet.
augmentation_demo = transforms.Compose([
    # Make the image a standard size.
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Always flip horizontally for demo so students can clearly see the effect.
    transforms.RandomHorizontalFlip(p=1.0),

    # Randomly rotate the image up to 15 degrees.
    transforms.RandomRotation(degrees=15),

    # Randomly change brightness, contrast, and saturation.
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),

    # Randomly crop and resize back to 224 x 224.
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),

    # Convert image to tensor so we can display transformed result.
    transforms.ToTensor(),
])

# Choose one class to demonstrate augmentation.
# Important: dataset class names use triple underscores.
sample_class = "Tomato___Early_blight"

# Pick the first image from this class.
sample_dir = os.path.join(TRAIN_DIR, sample_class)
sample_file = os.listdir(sample_dir)[0]
sample_path = os.path.join(sample_dir, sample_file)

# Open image using PIL and convert to RGB.
original_img = Image.open(sample_path).convert("RGB")

# Create a 2 x 4 grid of images.
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"Part 4 Data Augmentation Examples\nClass: {sample_class}", fontsize=13, fontweight="bold")

# Show the original image first.
axes[0, 0].imshow(original_img)
axes[0, 0].set_title("Original", fontsize=10, fontweight="bold")
axes[0, 0].axis("off")

# Labels for augmented examples.
aug_names = [
    "Augmented 1",
    "Augmented 2",
    "Augmented 3",
    "Augmented 4",
    "Augmented 5",
    "Augmented 6",
    "Augmented 7"
]

# Apply random augmentation multiple times to the same original image.
for i, name in enumerate(aug_names):
    row, col = divmod(i + 1, 4)
    ax = axes[row, col]

    # Every time this runs, a new random version is created.
    aug_tensor = augmentation_demo(original_img)

    # PyTorch image format is C x H x W.
    # Matplotlib expects H x W x C, so we rearrange dimensions.
    aug_img = aug_tensor.permute(1, 2, 0).numpy()

    # Keep values in the valid display range.
    aug_img = np.clip(aug_img, 0, 1)

    ax.imshow(aug_img)
    ax.set_title(name, fontsize=9)
    ax.axis("off")

plt.tight_layout()

# Save the visualization.
plt.savefig("part4_augmentation_examples.png", dpi=150)
plt.show()

print("Saved: part4_augmentation_examples.png")
```

## Explain After Cell 4

This cell shows what augmentation does visually.

Augmentation creates random variations like:

* flipping,
* rotating,
* cropping,
* zooming,
* brightness/contrast changes.

The label does not change. A rotated Early Blight leaf is still Early Blight.

## Ask Students

### Q1. What is data augmentation?

**Answer:**
Creating random modified versions of training images.

### Q2. Why is it useful?

**Answer:**
It helps the model handle real-world variation.

### Q3. Does augmentation create new image files?

**Answer:**
No. These random versions are generated while training.

---

# Cell 5 — Define Baseline and Augmented Transforms

```python
# Baseline transform: normal preprocessing only.
# No random augmentation here.
baseline_transform = transforms.Compose([
    # Resize all images to 224 x 224.
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Convert image to PyTorch tensor.
    transforms.ToTensor(),

    # Normalize using ImageNet mean and standard deviation.
    # This matches how ResNet18 was originally trained.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# Augmented transform: random changes applied to training images.
aug_transform = transforms.Compose([
    # Randomly crop/zoom the image and resize to 224 x 224.
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),

    # 50% chance to flip left-right.
    transforms.RandomHorizontalFlip(p=0.5),

    # 20% chance to flip up-down.
    transforms.RandomVerticalFlip(p=0.2),

    # Randomly rotate image up to 15 degrees.
    transforms.RandomRotation(degrees=15),

    # Slightly change brightness, contrast, saturation, and hue.
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),

    # Convert image to tensor.
    transforms.ToTensor(),

    # Normalize after augmentation.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# Validation transform: no augmentation.
# Validation images must stay fixed for fair comparison.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

print("Transforms ready.")
```

## Explain After Cell 5

There are three pipelines:

```text
baseline_transform → normal training images
aug_transform      → randomly changed training images
val_transform      → fixed validation images
```

Validation images are not augmented because validation should be stable and fair.

## Ask Students

### Q1. Why do we not augment validation images?

**Answer:**
Because validation is used as a fair test. If validation images keep changing, results are harder to compare.

### Q2. What does `p=0.5` mean in horizontal flip?

**Answer:**
Each image has a 50% chance of being flipped.

### Q3. Can augmentation be too strong?

**Answer:**
Yes. If the image becomes unrealistic, the model may learn bad patterns.

---

# Cell 6 — Load Datasets and DataLoaders

```python
print("Loading datasets...")

# Training dataset without augmentation.
# Uses the original images with only resize/tensor/normalize.
baseline_train = ImageFolder(root=TRAIN_DIR, transform=baseline_transform)

# DataLoader for baseline experiment.
baseline_loader = DataLoader(
    baseline_train,
    batch_size=BATCH_SIZE,
    shuffle=True,       # Shuffle training images.
    num_workers=2,      # Use background workers for loading images.
    pin_memory=True     # Helps transfer data to GPU faster.
)

# Training dataset with augmentation.
# Same images, but random changes are applied when loaded.
aug_train = ImageFolder(root=TRAIN_DIR, transform=aug_transform)

# DataLoader for augmentation experiment.
aug_loader = DataLoader(
    aug_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Validation dataset is the same for both experiments.
valid_dataset = ImageFolder(root=VALID_DIR, transform=val_transform)

# Validation loader does not need shuffling.
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Class names come from folder names.
CLASS_NAMES = baseline_train.classes
NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", NUM_CLASSES)
print("Training images:", len(baseline_train))
print("Validation images:", len(valid_dataset))
print("Baseline train batches:", len(baseline_loader))
print("Augmented train batches:", len(aug_loader))
print("Validation batches:", len(valid_loader))
```

## Explain After Cell 6

`baseline_train` and `aug_train` use the same folder. The difference is the transform.

Augmentation does not duplicate the dataset on disk. It changes images as they are loaded.

## Ask Students

### Q1. Are baseline and augmented datasets using different folders?

**Answer:**
No. They use the same images but different transforms.

### Q2. Why use the same validation loader?

**Answer:**
So both experiments are evaluated on the same test condition.

---

# Cell 7 — Helper Functions

```python
def build_model(num_classes, weights_path=None, device="cpu"):
    """
    Build ResNet18 with the correct final layer.
    Optionally load saved weights from Part 2.
    """

    # Build ResNet18 architecture.
    # weights=None because we are loading our saved Part 2 weights manually.
    model = models.resnet18(weights=None)

    # Replace final layer for plant disease classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load saved model weights if available.
    if weights_path is not None and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded weights from:", weights_path)
    else:
        print("No saved weights loaded. Starting from random weights.")

    # Move model to GPU or CPU.
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one full pass through the training dataset.
    """

    # Put model in training mode.
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    # Loop through training batches.
    for images, labels in loader:

        # Move images and labels to GPU or CPU.
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients before calculating new ones.
        optimizer.zero_grad()

        # Forward pass: model makes predictions.
        outputs = model(images)

        # Calculate mistake score.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Track total loss.
        total_loss += loss.item()

        # Pick highest scoring class as prediction.
        predicted = outputs.argmax(dim=1)

        # Count correct predictions.
        correct += (predicted == labels).sum().item()

        # Count total images seen.
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation data.
    No training happens here.
    """

    # Put model in evaluation mode.
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # No gradients needed because we are not updating the model.
    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only.
            outputs = model(images)

            # Calculate validation loss.
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
```

## Explain After Cell 7

This cell defines reusable functions.

* `build_model()` creates ResNet18 and loads saved Part 2 weights.
* `train_one_epoch()` trains the model.
* `evaluate()` checks validation performance without training.

## Ask Students

### Q1. Why do we load the same Part 2 weights for both experiments?

**Answer:**
To make the comparison fair.

### Q2. What is the difference between training and evaluation?

**Answer:**
Training updates the model. Evaluation only checks performance.

---

# Cell 8 — Experiment Function

```python
def run_experiment(name, train_loader, valid_loader, num_classes, num_epochs, device, weights_path=None):
    """
    Train one experiment and return training history.
    """

    print("\n" + "=" * 70)
    print("Experiment:", name)
    print("=" * 70)

    # Build model and optionally load Part 2 weights.
    model = build_model(
        num_classes=num_classes,
        weights_path=weights_path,
        device=device
    )

    # Loss function for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Optimizer updates model parameters.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler reduces learning rate if validation loss stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2
    )

    # Store results for plotting later.
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Track best validation accuracy for this experiment.
    best_val_acc = 0.0

    # Run training for selected number of epochs.
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train for one epoch.
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Evaluate on validation set.
        val_loss, val_acc = evaluate(
            model,
            valid_loader,
            criterion,
            device
        )

        # Adjust learning rate if needed.
        scheduler.step(val_loss)

        # Save results for graphs.
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # Save the best model for this experiment.
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Make a safe file name from experiment name.
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            model_path = f"resnet18_part4_{safe_name}_best.pth"

            torch.save(model.state_dict(), model_path)

            print(f"New best model saved: {model_path}")
            print(f"Best Val Acc: {best_val_acc:.2f}%")

    print("\nFinished experiment:", name)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return history, best_val_acc
```

## Explain After Cell 8

This function runs one full experiment. It lets us reuse the same logic for:

```text
Baseline No Aug
With Augmentation
```

It stores training and validation results so we can compare later.

## Ask Students

### Q1. What is an experiment here?

**Answer:**
One training run with a specific setup.

### Q2. Why save `history`?

**Answer:**
So we can plot accuracy and loss later.

---

# Cell 9 — Run Baseline Experiment

```python
# Experiment A: continue training without augmentation.
# This is the baseline/control experiment.
history_baseline, best_baseline = run_experiment(
    name="Baseline No Aug",
    train_loader=baseline_loader,
    valid_loader=valid_loader,
    num_classes=NUM_CLASSES,
    num_epochs=NUM_EPOCHS,
    device=device,
    weights_path=BASE_MODEL_PATH
)
```

## Explain After Cell 9

This is the control experiment.

It uses normal training images without augmentation.

## Ask Students

### Q1. What does baseline mean?

**Answer:**
The standard version used for comparison.

### Q2. Why do we need a baseline?

**Answer:**
So we can tell whether augmentation made things better or worse.

---

# Cell 10 — Run Augmentation Experiment

```python
# Experiment B: continue training with augmented images.
# This is the treatment experiment.
history_aug, best_aug = run_experiment(
    name="With Augmentation",
    train_loader=aug_loader,
    valid_loader=valid_loader,
    num_classes=NUM_CLASSES,
    num_epochs=NUM_EPOCHS,
    device=device,
    weights_path=BASE_MODEL_PATH
)
```

## Explain After Cell 10

This experiment uses randomly changed training images.

Everything else should be the same as the baseline experiment.

## Ask Students

### Q1. What changed in this experiment?

**Answer:**
The training images were augmented.

### Q2. What stayed the same?

**Answer:**
Starting model, validation set, learning rate, epochs, and model architecture.

---

# Cell 11 — Compare Validation Accuracy and Loss

```python
epochs = range(1, NUM_EPOCHS + 1)

# Plot validation accuracy comparison.
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history_baseline["val_acc"],
    marker="o",
    label=f"No Augmentation best: {best_baseline:.1f}%"
)

plt.plot(
    epochs,
    history_aug["val_acc"],
    marker="o",
    label=f"With Augmentation best: {best_aug:.1f}%"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Part 4 Augmentation Experiment — Validation Accuracy")
plt.ylim(0, 105)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("part4_augmentation_accuracy.png", dpi=150)
plt.show()

# Plot validation loss comparison.
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history_baseline["val_loss"],
    marker="o",
    label="No Augmentation"
)

plt.plot(
    epochs,
    history_aug["val_loss"],
    marker="o",
    label="With Augmentation"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Part 4 Augmentation Experiment — Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("part4_augmentation_loss.png", dpi=150)
plt.show()

print("Saved:")
print("part4_augmentation_accuracy.png")
print("part4_augmentation_loss.png")
```

## Explain After Cell 11

This cell compares both experiments visually.

Validation accuracy should ideally go up.

Validation loss should ideally go down.

## Ask Students

### Q1. Which experiment has better validation accuracy?

**Answer:**
Students should read the graph.

### Q2. Which experiment has lower validation loss?

**Answer:**
Students should read the graph.

---

# Cell 12 — Print Comparison Summary

```python
# Difference between augmentation and baseline.
diff = best_aug - best_baseline

print("=" * 60)
print("Part 4 Experiment Summary")
print("=" * 60)

print(f"{'Metric':<35} {'No Aug':>12} {'With Aug':>12}")
print("-" * 60)

print(f"{'Best Val Accuracy':<35} {best_baseline:>11.2f}% {best_aug:>11.2f}%")

print(f"{'Final Epoch Val Acc':<35} "
      f"{history_baseline['val_acc'][-1]:>11.2f}% "
      f"{history_aug['val_acc'][-1]:>11.2f}%")

print(f"{'Improvement from Augmentation':<35} {diff:>+23.2f}%")

print("=" * 60)

# Interpret result.
if diff > 0:
    print(f"Augmentation improved validation accuracy by {diff:.2f}%.")
elif diff == 0:
    print("Augmentation gave the same best validation accuracy.")
else:
    print(f"Augmentation reduced validation accuracy by {diff:.2f}%.")
    print("That can happen if augmentation is too strong or if more epochs are needed.")
```

## Explain After Cell 12

This gives the numerical result of the experiment.

Positive improvement means augmentation helped.

Negative improvement means augmentation did not help in this run.

## Ask Students

### Q1. Did augmentation help?

**Answer:**
Check whether the improvement number is positive.

### Q2. If augmentation reduced accuracy, is augmentation always bad?

**Answer:**
No. It may need different settings, fewer transformations, or more training time.

---

# Cell 13 — Overfitting Check

```python
print("Overfitting Check")
print("=" * 60)

for name, history in [
    ("No Augmentation", history_baseline),
    ("With Augmentation", history_aug)
]:
    # Final training accuracy.
    train_final = history["train_acc"][-1]

    # Final validation accuracy.
    val_final = history["val_acc"][-1]

    # Gap between training and validation performance.
    gap = train_final - val_final

    print("\n", name)
    print("Train Acc final:", f"{train_final:.2f}%")
    print("Val Acc final:  ", f"{val_final:.2f}%")
    print("Gap:            ", f"{gap:.2f}%")

    # Large gap may indicate overfitting.
    if gap > 10:
        print("Possible overfitting: training accuracy is much higher than validation accuracy.")
    else:
        print("Gap looks reasonable.")
```

## Explain After Cell 13

Overfitting means the model does very well on training data but worse on validation data.

A large train-validation gap can suggest overfitting.

Augmentation can reduce overfitting by making training examples more varied.

## Ask Students

### Q1. What is overfitting?

**Answer:**
The model memorizes training images but does not generalize well to new images.

### Q2. How can augmentation help?

**Answer:**
It gives the model more varied examples, making memorization harder.

---

# Cell 14 — Check Output Files

```python
files_to_check = [
    "part4_augmentation_examples.png",
    "part4_augmentation_accuracy.png",
    "part4_augmentation_loss.png",
    "resnet18_part4_baseline_no_aug_best.pth",
    "resnet18_part4_with_augmentation_best.pth"
]

# Check whether all expected output files were created.
for f in files_to_check:
    print(f, "exists:", os.path.exists(f))
```

## Explain After Cell 14

This checks whether the notebook created the expected outputs.

These files can be used later in final reports or presentations.

## Ask Students

### Q1. Which files are useful for presentations?

**Answer:**
The augmentation example image, accuracy graph, loss graph, and experiment summary.

---

# Part 4 Closing Summary

Use this at the end:

> In Part 4, we ran a real machine learning experiment. We compared a baseline model trained normally with a model trained using augmented images. We changed one major thing: the training image pipeline. Then we measured validation accuracy and validation loss to see whether augmentation helped.

---

# Final Discussion Questions

## Q1. What is data augmentation?

**Answer:**
Creating random modified versions of training images.

## Q2. Why does augmentation help?

**Answer:**
It helps the model handle real-world variation such as different angles, lighting, and zoom.

## Q3. Why do we not augment validation images?

**Answer:**
Validation must stay fixed so results are fair and comparable.

## Q4. What is a baseline experiment?

**Answer:**
The standard setup used for comparison.

## Q5. What is the treatment experiment?

**Answer:**
The changed setup — here, training with augmentation.

## Q6. What does overfitting mean?

**Answer:**
The model performs very well on training data but not as well on new data.

## Q7. How can augmentation reduce overfitting?

**Answer:**
It makes training examples more varied, so the model is less likely to memorize.

## Q8. What if augmentation makes accuracy worse?

**Answer:**
It may be too strong, the model may need more epochs, or the original model may already be strong.

## Q9. Why is this a real experiment?

**Answer:**
Because we compare a control setup against a changed setup and measure the result.

## Q10. What could we try next?

**Answer:**
Different rotation amounts, different crop settings, different brightness changes, more epochs, or class-specific analysis.
