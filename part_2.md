# Plant Disease Detection with AI — Part 2 Instructor Monitor

## Part 2: Train a Plant Disease Classifier with Transfer Learning

In Part 1, students explored the dataset.

In Part 2, students train a real image classification model using **Transfer Learning** with **ResNet18**. The original notebook uses a pretrained ResNet18 model, replaces its final layer for plant disease classes, trains it, saves the best model, and plots learning curves.

---

# Big Picture Explanation

Use this before starting the notebook:

> Today we are training a plant disease classifier. Instead of building an AI model from zero, we are using a model that already learned from millions of images. We will replace its final answer layer so it predicts plant disease classes instead of general objects like cats, dogs, and cars.

Simple flow:

```text
Load plant images
        ↓
Prepare images for ResNet18
        ↓
Load pretrained ResNet18
        ↓
Replace final layer
        ↓
Train on plant disease images
        ↓
Check validation accuracy
        ↓
Save the best model
        ↓
Plot learning curves
```

Main idea:

> We borrow a model that already knows general image patterns and teach it our plant disease labels.

---

# Cell 1 — Imports and GPU Check

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

## Explain After Cell 1

This cell loads the Python libraries needed for training.

Important terms:

* `torch`: main PyTorch library.
* `torchvision`: image datasets, transforms, and pretrained models.
* `DataLoader`: sends images to the model in batches.
* `matplotlib`: used for plotting learning curves.
* `CUDA`: technology that lets PyTorch use the GPU.

## Ask Students

### Q1. Why do we want to use a GPU?

**Answer:**
Training a deep learning model involves many calculations. A GPU can do many of those calculations in parallel, so training is much faster.

### Q2. Are we guaranteed to always have a GPU?

**Answer:**
No. The code checks whether a GPU is available. If not, it uses the CPU.

### Q3. What does `torch.manual_seed(42)` do?

**Answer:**
It helps make results more repeatable by controlling some randomness in PyTorch.
Ex: Why it matters for weights?: Neural networks start by picking "random" numbers for their weights. If you do not set a seed, the computer picks a random page every time you hit run. By setting the seed, you force the computer to pick the exact same starting weights every single time.

---

# Cell 2 — Dataset Path

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

## Explain After Cell 2

The dataset has two important folders:

```text
train/
valid/
```

Training images are used for learning.

Validation images are used to check whether the model works on images it is not directly learning from.

## Ask Students

### Q1. What is the difference between `train` and `valid`?

**Answer:**
`train` contains images the model learns from. `valid` contains separate images used to check how well the model performs.

### Q2. Why should validation images be separate from training images?

**Answer:**
Because we want to know whether the model learned general patterns, not just memorized the training images.

---

# Cell 3 — Image Transforms and Datasets

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

## Explain After Cell 3

Before images go into the model, they must be prepared.

The transform does three things:

```text
resize → tensor → normalize
```

### Resize

All images become:

```text
224 × 224
```

because ResNet18 expects that size.

### Tensor

A tensor is the number format PyTorch uses.

An image becomes something like:

```text
3 × 224 × 224
```

meaning:

```text
3 color channels
224 height
224 width
```

### Normalize

Normalization adjusts image numbers into the format ResNet18 expects.

Because ResNet18 was trained on ImageNet, we use ImageNet normalization values.

## Ask Students

### Q1. Why do we resize all images?

**Answer:**
The model expects images to have the same size. Resizing makes every image consistent.

### Q2. What is a tensor?

**Answer:**
A tensor is the numerical format PyTorch uses to store and process data.

### Q3. Why do we normalize the images?

**Answer:**
Because the pretrained ResNet18 model expects images to be prepared in the same way as the images it originally learned from.

### Q4. Where do the class labels come from?

**Answer:**
They come from the folder names. `ImageFolder` uses each folder name as a class label.

---

# Cell 4 — DataLoaders

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

## Explain After Cell 4

A `DataLoader` feeds images to the model in small groups called **batches**.

Here:

```text
BATCH_SIZE = 32
```

means the model sees 32 images at a time.

We shuffle training images so the model does not learn from folder order.

We do not need to shuffle validation images because validation is just checking performance.

## Ask Students

### Q1. What is a batch?

**Answer:**
A batch is a small group of images processed together.

### Q2. Why not process the whole dataset at once?

**Answer:**
The dataset may be too large for memory, and training works better by updating the model after small groups.

### Q3. Why do we shuffle training images?

**Answer:**
So the model sees a mixed order of examples instead of learning based on folder order.

---

# Cell 5 — Load ResNet18 and Replace Final Layer

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

## Explain After Cell 5

This is the main transfer learning step.

ResNet18 was originally trained to predict 1000 ImageNet classes.

Examples:

```text
cat
dog
car
airplane
chair
```

But we need plant disease classes.

So we replace the final layer:

```text
Old final layer: 512 features → 1000 ImageNet classes
New final layer: 512 features → 38 plant disease classes
```

Important clarification:

> Setting the output size to 38 does not magically teach the model plant diseases. It only creates 38 output slots. The model learns what those slots mean from the labeled training images.

Simple analogy:

> We keep the model’s eyes, but replace its answer sheet.

## Ask Students

### Q1. What is transfer learning?

**Answer:**
Transfer learning means using a model that already learned from one task and adapting it to a new task.

### Q2. Why use ResNet18 instead of training from zero?

**Answer:**
ResNet18 already learned useful image patterns like edges, shapes, textures, and colors. That helps it learn plant diseases faster.

### Q3. What does replacing the final layer do?

**Answer:**
It changes the model’s output from 1000 ImageNet classes to our plant disease classes.

### Q4. Does the number 38 teach the model what plant diseases are?

**Answer:**
No. It only tells the model there are 38 possible outputs. The labels in the dataset teach the model what each output means.

---

# Cell 6 — Loss Function, Optimizer, and Scheduler

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

## Explain After Cell 6

This cell defines how the model learns.

### Loss

Loss is the model’s mistake score.

```text
High loss = worse
Low loss = better
```

### Optimizer

The optimizer updates model weights after mistakes.

### Learning Rate

The learning rate controls how big each update step is.

```text
Gradient = direction
Learning rate = step size
Optimizer = takes the step
Loss = mistake score
```

### Scheduler

The scheduler reduces the learning rate if validation loss stops improving.

## Ask Students

### Q1. What is loss?

**Answer:**
Loss is a number that tells how wrong the model’s prediction was.

### Q2. What does the optimizer do?

**Answer:**
It updates the model’s internal weights so the model can improve.

### Q3. What is learning rate?

**Answer:**
It controls how big the model’s update steps are.

### Q4. What happens if learning rate is too high?

**Answer:**
The model may jump around and not learn well.

### Q5. What happens if learning rate is too low?

**Answer:**
The model may learn very slowly.

---

# Cell 7 — Training and Evaluation Functions

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

## Explain After Cell 7

This cell defines two functions:

```text
train_one_epoch()
evaluate()
```

### `train_one_epoch()`

This is for training.

It does:

```text
make prediction
calculate loss
calculate gradients
update weights
track accuracy
```

### `evaluate()`

This is for validation.

It does:

```text
make prediction
calculate loss
track accuracy
```

But it does **not** update the model.

Important distinction:

```text
Training = practice and improve
Validation = quiz and check
```

## Ask Students

### Q1. What is an epoch?

**Answer:**
One epoch means the model has gone through the full training dataset once.

### Q2. What does `model.train()` mean?

**Answer:**
It puts the model in training mode.

### Q3. What does `model.eval()` mean?

**Answer:**
It puts the model in evaluation mode.

### Q4. Why do we use `torch.no_grad()` during evaluation?

**Answer:**
Because we are only checking performance, not updating the model.

### Q5. What does `outputs.argmax(dim=1)` do?

**Answer:**
It picks the class with the highest score for each image.

### Q6. Why do we use `optimizer.zero_grad()`?

**Answer:**
PyTorch accumulates gradients by default, so we clear old gradients before calculating new ones.

---

# Cell 8 — Train the Model

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

## Explain After Cell 8

This is where training actually happens.

Each epoch does:

```text
train on training images
evaluate on validation images
save results
save best model
```

The model is saved only if validation accuracy improves.

Why validation accuracy?

Because it shows how well the model works on images it did not directly learn from.

## Ask Students

### Q1. Why do we save the best model instead of just the last model?

**Answer:**
The last model is not always the best. We save the model with the highest validation accuracy.

### Q2. Why is validation accuracy important?

**Answer:**
It shows how well the model performs on separate images, not just the ones it trained on.

### Q3. What does training accuracy tell us?

**Answer:**
How well the model performs on images it is learning from.

### Q4. What could it mean if training accuracy is high but validation accuracy is low?

**Answer:**
The model may be overfitting, meaning it memorized training images but does not generalize well.

---

# Cell 9 — Check Saved Model

```python
# Check whether the trained model file was created.
print("Model saved:", os.path.exists("resnet18_plantdisease_part2_best.pth"))
```

Expected output:

```text
Model saved: True
```

## Explain After Cell 9

This checks whether the trained model file exists.

The saved file contains the learned model weights.

## Ask Students

### Q1. What does the `.pth` file store?

**Answer:**
It stores the trained model weights.

### Q2. Why save the model?

**Answer:**
So we can use it later without training again.

---

# Cell 10 — Plot Learning Curves

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

## Explain After Cell 10

Learning curves show what happened during training.

There are two types:

```text
loss curve
accuracy curve
```

Good signs:

```text
loss goes down
accuracy goes up
```

But with only 1 epoch, the graph is just a quick test. It becomes more useful with more epochs.

## Ask Students

### Q1. What should happen to loss if the model is learning?

**Answer:**
Loss should generally go down.

### Q2. What should happen to accuracy if the model is learning?

**Answer:**
Accuracy should generally go up.

### Q3. What does it mean if training accuracy rises but validation accuracy does not?

**Answer:**
The model may be overfitting.

---

# Cell 11 — Check Output Files

```python
files_to_check = [
    "resnet18_plantdisease_part2_best.pth",
    "part2_loss_curve.png",
    "part2_accuracy_curve.png"
]

for f in files_to_check:
    print(f, "exists:", os.path.exists(f))
```

## Explain After Cell 11

This confirms the model and plots were saved.

These files will be useful in later parts:

```text
Part 3: load saved model and evaluate it
Part 4: compare training experiments
```

## Ask Students

### Q1. Why do we need the saved model for the next part?

**Answer:**
Part 3 evaluates the trained model. Without the saved model, we would need to train again.

---

# Optional — Run Full Training

After the 1-epoch test works, rerun training with:

```python
NUM_EPOCHS = 5
```

Then rerun:

```text
Cell 8
Cell 10
Cell 11
```

## Explain Before Full Training

The 1-epoch run is only a test.

A full run gives the model more practice.

## Ask Students

### Q1. What does increasing epochs do?

**Answer:**
It lets the model go through the full training dataset more times.

### Q2. Is more epochs always better?

**Answer:**
No. Too many epochs can cause overfitting.

---

# Part 2 Closing Summary

Use this at the end:

> In Part 2, we trained a plant disease classifier using transfer learning. We used ResNet18, a model that already knows general image patterns. We replaced its final layer so it could predict plant disease classes. Then we trained it using labeled plant images, checked validation accuracy, saved the best model, and plotted learning curves.

---

# Final Discussion Questions

## Q1. What is transfer learning?

**Answer:**
Using a model trained on one task and adapting it to a new task.

## Q2. Why did we replace the final layer?

**Answer:**
The original model predicted 1000 ImageNet classes. We needed it to predict plant disease classes instead.

## Q3. Does the model know the class names directly?

**Answer:**
No. The model works with class numbers. The folder names map those numbers to class names.

## Q4. What is loss?

**Answer:**
Loss is the model’s mistake score.

## Q5. What is the optimizer doing?

**Answer:**
It updates the model weights to reduce mistakes.

## Q6. Why do we use validation data?

**Answer:**
To check whether the model works on images it did not directly train on.

## Q7. What does saving the best model mean?

**Answer:**
It saves the model weights from the epoch with the best validation accuracy.
