# Plant Disease Detection with AI — Part 3 Instructor Monitor

## Part 3: Understanding Your Model — What Did It Learn?

In Part 2, we trained a ResNet18 model using transfer learning.

In Part 3, we evaluate that trained model.

The goal is to help students understand:

* How well the model performs.
* Which classes it predicts correctly.
* Which classes it confuses.
* What precision, recall, F1-score, and support mean.
* Why accuracy alone is not enough.
* How to visually inspect correct and wrong predictions.

---

# Big Picture Explanation

Use this before starting the notebook:

> In Part 2, we trained the model. Today, we are not training from scratch. We are checking what the model learned. We will load the saved model, run it on validation images, compare its predictions with the correct labels, and study where it makes mistakes.

Simple flow:

```text
Load validation images
        ↓
Load trained model
        ↓
Run model on validation images
        ↓
Compare predictions with true labels
        ↓
Build confusion matrix
        ↓
Study correct and wrong predictions
        ↓
Read precision, recall, F1-score, and support
```

---

# Cell 1 — Imports and Device Check

```python
import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds to make results more repeatable.
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Use GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Explain After Cell 1

We are importing the tools needed for evaluation.

Important packages:

* `torch`: loads and runs the trained model.
* `torchvision`: loads image datasets and ResNet18.
* `matplotlib`: plots images and charts.
* `seaborn`: makes the confusion matrix easier to view.
* `sklearn.metrics`: gives confusion matrix and classification report.

## Ask Students

### Q1. Are we training a new model in this part?

**Answer:**
No. We are loading the model trained in Part 2 and evaluating it.

### Q2. Why do we check for GPU again?

**Answer:**
Because running predictions on many images is faster on GPU.

---

# Cell 2 — Dataset and Model Paths

```python
DATASET_PATH = "/fs/ess/PAS2699/osc_summer_camp_2026/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

# Validation images are used to test the model.
VALID_DIR = os.path.join(DATASET_PATH, "valid")

# Training directory is included in case we need class names or extra checks.
TRAIN_DIR = os.path.join(DATASET_PATH, "train")

# This is the model saved from Part 2.
MODEL_PATH = "resnet18_plantdisease_part2_best.pth"

print("Valid exists:", os.path.exists(VALID_DIR))
print("Train exists:", os.path.exists(TRAIN_DIR))
print("Model exists:", os.path.exists(MODEL_PATH))
```

Expected output:

```text
Valid exists: True
Train exists: True
Model exists: True
```

## Explain After Cell 2

We need two things:

1. The validation images.
2. The saved model from Part 2.

The validation folder contains images the model did not directly train on. That makes it useful for checking whether the model learned general patterns.

## Ask Students

### Q1. Why do we use validation images instead of training images?

**Answer:**
Training images are the examples the model learned from. Validation images are separate examples used to check whether the model can handle new images.

### Q2. What does the `.pth` file contain?

**Answer:**
It contains the trained model weights, meaning the values the model learned during training.

---

# Cell 3 — Load Validation Dataset

```python
IMG_SIZE = 224
BATCH_SIZE = 32

# Validation images must be prepared the same way as in Part 2.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ImageFolder reads folders as class labels.
valid_dataset = ImageFolder(root=VALID_DIR, transform=val_transform)

# DataLoader sends images to the model in batches.
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Class names come from folder names.
CLASS_NAMES = valid_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print("Validation images:", len(valid_dataset))
print("Number of classes:", NUM_CLASSES)

print("\nClasses:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{i:>2}: {cls}")
```

## Explain After Cell 3

We are loading only the validation dataset.

The validation images are processed the same way as during training:

```text
resize → tensor → normalize
```

This is important because the model expects the same input format it saw during Part 2.

## Ask Students

### Q1. Why do we resize and normalize validation images too?

**Answer:**
Because the trained model expects images in the same format used during training.

### Q2. Where do the class names come from?

**Answer:**
They come from the folder names inside the validation dataset.

---

# Cell 4 — Load the Trained Part 2 Model

```python
# Build the same ResNet18 structure used in Part 2.
model = models.resnet18(weights=None)

# Replace the final layer with the correct number of plant disease classes.
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load the learned weights saved from Part 2.
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Move model to GPU or CPU.
model = model.to(device)

# Put model in evaluation mode.
model.eval()

print("Model loaded successfully.")
```

## Explain After Cell 4

We recreate the same model structure used in Part 2:

```text
ResNet18 + final layer for plant disease classes
```

Then we load the trained weights from the `.pth` file.

Important:

> The `.pth` file stores the model’s learned values, not the full Python notebook.

We use:

```python
model.eval()
```

because we are evaluating, not training.

## Ask Students

### Q1. Why do we build ResNet18 again before loading the saved file?

**Answer:**
Because the saved file contains learned weights. We still need to recreate the model architecture before loading those weights.

### Q2. Why do we use `model.eval()`?

**Answer:**
It tells PyTorch that we are testing/evaluating the model, not training it.

---

# Cell 5 — Run Predictions on All Validation Images

```python
all_preds = []
all_labels = []
all_probs = []

print("Running model on validation images...")

# no_grad means we are not training or calculating gradients.
with torch.no_grad():
    for images, labels in valid_loader:

        # Move images to GPU or CPU.
        images = images.to(device)

        # Model gives raw scores for each class.
        outputs = model(images)

        # Convert raw scores into probabilities.
        probs = torch.softmax(outputs, dim=1)

        # Pick the class with the highest probability.
        predicted = probs.argmax(dim=1)

        # Store predictions, true labels, and probabilities.
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert lists to NumPy arrays for easier analysis.
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Overall accuracy = correct predictions / total predictions.
overall_acc = 100 * (all_preds == all_labels).mean()

print("Done.")
print(f"Overall validation accuracy: {overall_acc:.2f}%")
```

## Explain After Cell 5

This cell runs every validation image through the model.

For each image, we store:

```text
true label
predicted label
prediction probabilities
```

The model first produces raw scores. Then:

```python
torch.softmax(outputs, dim=1)
```

turns those scores into probabilities.

`dim=1` means:

> Apply softmax across the class scores for each image.

If the output shape is:

```text
[batch_size, num_classes]
```

then `dim=1` is the class dimension.

## Ask Students

### Q1. What does the model output for each image?

**Answer:**
It outputs one score for each possible class.

### Q2. What does softmax do?

**Answer:**
It converts raw scores into probabilities that add up to 1.

### Q3. What does `argmax` do?

**Answer:**
It chooses the class with the highest probability.

### Q4. What does overall accuracy mean?

**Answer:**
It is the percentage of validation images the model predicted correctly.

---

# Cell 6 — Build Confusion Matrix

```python
# Confusion matrix compares true labels with predicted labels.
cm = confusion_matrix(all_labels, all_preds)

# Shorten class names for easier plotting.
short_names = [c.split("__")[-1][:15] for c in CLASS_NAMES]

plt.figure(figsize=(20, 18))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=short_names,
    yticklabels=short_names,
    linewidths=0.5
)

plt.title("Part 3 Confusion Matrix\nRows = Actual Class, Columns = Predicted Class")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig("part3_confusion_matrix.png", dpi=150)
plt.show()

print("Saved: part3_confusion_matrix.png")
```

## Explain After Cell 6

The confusion matrix shows where the model is right and wrong.

Rows are:

```text
actual class
```

Columns are:

```text
predicted class
```

A perfect model would have most values on the diagonal.

Mistakes appear away from the diagonal.

## Ask Students

### Q1. What does a diagonal value mean?

**Answer:**
The model predicted the correct class.

### Q2. What does an off-diagonal value mean?

**Answer:**
The model confused one class with another.

### Q3. Why is a confusion matrix better than only accuracy?

**Answer:**
Accuracy gives one number, but the confusion matrix shows exactly which classes are being confused.

---

# Cell 7 — Find Easiest and Hardest Classes

```python
per_class_acc = []

# Calculate accuracy separately for each class.
for i in range(NUM_CLASSES):

    # Select only images whose real label is class i.
    mask = (all_labels == i)

    if mask.sum() > 0:
        # Among real class-i images, check how many were predicted correctly.
        acc = 100 * (all_preds[mask] == i).mean()
    else:
        acc = 0.0

    per_class_acc.append(acc)

# Sort classes from lowest accuracy to highest accuracy.
sorted_indices = np.argsort(per_class_acc)

print("Top 5 easiest classes:")
for i in sorted_indices[-5:][::-1]:
    print(f"{CLASS_NAMES[i]:<50} {per_class_acc[i]:.2f}%")

print("\nBottom 5 hardest classes:")
for i in sorted_indices[:5]:
    print(f"{CLASS_NAMES[i]:<50} {per_class_acc[i]:.2f}%")
```

## Explain After Cell 7

Overall accuracy can hide weak classes.

This cell checks accuracy class by class.

A model may have high overall accuracy but still perform badly on one disease class.

## Ask Students

### Q1. Why do we calculate accuracy for each class?

**Answer:**
Because the model may be strong on some diseases and weak on others.

### Q2. What should we do with the hardest classes?

**Answer:**
Look at their images, check if they look similar to other classes, and consider collecting more data or using augmentation.

---

# Cell 8 — Plot Per-Class Accuracy

```python
plt.figure(figsize=(20, 6))

plt.bar(range(NUM_CLASSES), per_class_acc)

# Overall accuracy line for comparison.
plt.axhline(
    y=overall_acc,
    linestyle="--",
    label=f"Overall Accuracy: {overall_acc:.1f}%"
)

plt.xticks(range(NUM_CLASSES), short_names, rotation=45, ha="right", fontsize=7)
plt.ylabel("Accuracy (%)")
plt.title("Part 3 Accuracy by Disease Class")
plt.legend()
plt.tight_layout()
plt.savefig("part3_class_accuracies.png", dpi=150)
plt.show()

print("Saved: part3_class_accuracies.png")
```

## Explain After Cell 8

This chart shows which disease classes are easier or harder for the model.

The dashed line is the overall accuracy.

Bars below the line are weaker than average.

Bars above the line are stronger than average.

## Ask Students

### Q1. What does a low bar mean?

**Answer:**
The model struggles with that class.

### Q2. What does the dashed line show?

**Answer:**
The overall validation accuracy.

---

# Cell 9 — Helper Function to Unnormalize Images

```python
def unnormalize(tensor):
    """
    Convert a normalized image tensor back into displayable image format.
    """

    # These are the same ImageNet mean and std values used during normalization.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Reverse normalization.
    img = tensor.cpu() * std + mean

    # Keep values between 0 and 1 so matplotlib can display them.
    img = img.clamp(0, 1)

    # Convert from C x H x W to H x W x C for matplotlib.
    return img.permute(1, 2, 0).numpy()
```

## Explain After Cell 9

The model sees normalized image tensors.

But to display images nicely, we must reverse normalization.

This function converts the image back into a format that `matplotlib` can show.

## Ask Students

### Q1. Why do images need to be unnormalized before display?

**Answer:**
Because normalized tensors are formatted for the model, not for human viewing.

### Q2. Why do we change `C x H x W` to `H x W x C`?

**Answer:**
PyTorch stores images as channels-first, but Matplotlib expects channels-last.

---

# Cell 10 — Function to Show Predictions

```python
def show_predictions(dataset, model, device, class_names, n=8, only_wrong=False):
    """
    Show random model predictions.
    If only_wrong=True, show only misclassified examples.
    """

    # Randomize image order.
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    results = []

    for idx in indices:
        image, true_label = dataset[idx]

        # Add batch dimension: [3, 224, 224] becomes [1, 3, 224, 224].
        image_batch = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_batch)
            probs = torch.softmax(output, dim=1)

            # Predicted class.
            pred = probs.argmax(dim=1).item()

            # Confidence of predicted class.
            conf = probs[0, pred].item() * 100

        is_wrong = (pred != true_label)

        # If we only want wrong predictions, skip correct ones.
        if only_wrong and not is_wrong:
            continue

        results.append((image, true_label, pred, conf, is_wrong))

        if len(results) >= n:
            break

    if len(results) == 0:
        print("No examples found.")
        return

    cols = 4
    rows = int(np.ceil(len(results) / cols))

    title = "Wrong Predictions" if only_wrong else "Random Predictions"

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, (image, true_lbl, pred_lbl, conf, wrong) in enumerate(results):
        ax = axes[i]
        ax.imshow(unnormalize(image))
        ax.axis("off")

        true_short = class_names[true_lbl].split("__")[-1]
        pred_short = class_names[pred_lbl].split("__")[-1]

        status = "WRONG" if wrong else "RIGHT"

        ax.set_title(
            f"{status}\nTrue: {true_short}\nPred: {pred_short}\nConf: {conf:.1f}%",
            fontsize=8
        )

    plt.tight_layout()

    filename = "part3_predictions_wrong.png" if only_wrong else "part3_predictions_random.png"
    plt.savefig(filename, dpi=150)
    plt.show()

    print("Saved:", filename)
```

## Explain After Cell 10

This function lets us visually inspect predictions.

It shows:

```text
image
true label
predicted label
confidence
right or wrong
```

This is important because numbers alone do not explain everything.

Sometimes the model’s mistake may be understandable if two diseases look visually similar.

## Ask Students

### Q1. Why is it useful to look at actual images after seeing metrics?

**Answer:**
Because images help us understand why the model made certain mistakes.

### Q2. What does confidence mean here?

**Answer:**
It is the probability assigned to the predicted class.

### Q3. Can a model be confidently wrong?

**Answer:**
Yes. A model can assign high confidence to the wrong class.

---

# Cell 11 — Show Random Predictions

```python
show_predictions(
    dataset=valid_dataset,
    model=model,
    device=device,
    class_names=CLASS_NAMES,
    n=8,
    only_wrong=False
)
```

## Explain After Cell 11

This shows a random mix of correct and incorrect predictions.

Students should compare the true label and predicted label.

## Ask Students

### Q1. Are the predictions mostly correct?

**Answer:**
Depends on the trained model accuracy. Students should inspect the displayed examples.

### Q2. Do the confidence values seem reasonable?

**Answer:**
High confidence means the model strongly preferred that class, but it can still be wrong.

---

# Cell 12 — Show Wrong Predictions Only

```python
show_predictions(
    dataset=valid_dataset,
    model=model,
    device=device,
    class_names=CLASS_NAMES,
    n=8,
    only_wrong=True
)
```

## Explain After Cell 12

Wrong predictions are very useful.

They help us ask:

```text
Did the model make an obvious mistake?
Are the diseases visually similar?
Is the image blurry or unclear?
Could the label be difficult even for humans?
```

## Ask Students

### Q1. Why do we intentionally look at wrong predictions?

**Answer:**
Because mistakes show us where the model needs improvement.

### Q2. If two diseases look similar, what might happen?

**Answer:**
The model may confuse them.

### Q3. What could help reduce these mistakes?

**Answer:**
More data, better image quality, data augmentation, or more training.

---

# Cell 13 — Classification Report

```python
print("Classification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[c.split("__")[-1] for c in CLASS_NAMES],
        digits=2
    )
)
```

## Explain After Cell 13

The classification report gives more detailed metrics for each class:

```text
precision
recall
f1-score
support
```

### Precision

Precision asks:

```text
When the model predicts this class, how often is it right?
```

### Recall

Recall asks:

```text
Of all the real examples of this class, how many did the model catch?
```

### F1-score

F1-score balances precision and recall.

# Precision, Recall, F1-Score — Simple Explanation

## 1. Precision

**Precision asks:**

> When the model predicts a class, how often is it correct?

Example:

Suppose the model predicts **Tomato Early Blight** for 80 images.

Out of those 80 predictions:

```text
60 are actually Tomato Early Blight
20 are not Tomato Early Blight
```

So:

```text
Precision = correct predictions for this class / total predictions for this class
Precision = 60 / 80
Precision = 0.75 = 75%
```

Meaning:

> When the model says “Tomato Early Blight,” it is correct 75% of the time.

High precision means the model’s prediction is trustworthy.

---

## 2. Recall

**Recall asks:**

> Out of all real examples of a class, how many did the model catch?

Example:

Suppose there are 100 actual **Tomato Early Blight** images in the validation set.

The model correctly finds 60 of them.

So:

```text
Recall = correctly found examples / total real examples
Recall = 60 / 100
Recall = 0.60 = 60%
```

Meaning:

> The model found 60% of the real Tomato Early Blight images, but missed 40%.

High recall means the model catches most real cases.

---

## 3. Precision vs Recall Example

Suppose we are detecting **Tomato Early Blight**.

There are:

```text
100 real Tomato Early Blight images
900 images from other classes
```

The model predicts **Tomato Early Blight** for 300 images.

Out of those 300 predictions:

```text
95 are actually Tomato Early Blight
205 are not Tomato Early Blight
```

So:

```text
Precision = 95 / 300 = 31.7%
Recall    = 95 / 100 = 95%
```

This means:

```text
High recall    → The model caught most real Early Blight cases.
Low precision  → The model also created many false alarms.
```

Simple interpretation:

> The model is very aggressive. It catches most real disease cases, but it also wrongly labels many other leaves as that disease.

---

## 4. Opposite Example: High Precision, Low Recall

Suppose there are again:

```text
100 real Tomato Early Blight images
```

The model predicts **Tomato Early Blight** only 20 times.

Out of those 20 predictions:

```text
18 are actually Tomato Early Blight
2 are not Tomato Early Blight
```

So:

```text
Precision = 18 / 20 = 90%
Recall    = 18 / 100 = 18%
```

This means:

```text
High precision → When the model says Early Blight, it is usually correct.
Low recall     → But it misses most real Early Blight cases.
```

Simple interpretation:

> The model is very cautious. It avoids false alarms, but it misses many actual disease cases.

---

## 5. F1-Score

**F1-score combines precision and recall into one balanced score.**

It is useful when we care about both:

```text
not making too many false alarms
not missing too many real disease cases
```

Formula:

```text
F1 = 2 × (precision × recall) / (precision + recall)
```

The F1-score is high only when both precision and recall are reasonably high.

---

## 6. How F1-Score Changes

### Case A — Both precision and recall are high

```text
Precision = 0.90
Recall    = 0.90
F1-score  = 0.90
```

Meaning:

> The model is both trustworthy and good at finding real cases.

This is ideal.

---

### Case B — Precision high, recall low

```text
Precision = 0.90
Recall    = 0.18
F1-score  ≈ 0.30
```

Meaning:

> The model is correct when it predicts the disease, but it misses many real disease cases.

F1-score becomes low because recall is low.

---

### Case C — Precision low, recall high

```text
Precision = 0.32
Recall    = 0.95
F1-score  ≈ 0.48
```

Meaning:

> The model catches most real disease cases, but it creates many false alarms.

F1-score is not very high because precision is low.

---

### Case D — Both precision and recall are low

```text
Precision = 0.30
Recall    = 0.30
F1-score  = 0.30
```

Meaning:

> The model is weak overall for that class.

---

## 7. Simple Way to Explain F1-Score

Use this:

```text
Precision = Can I trust the model when it predicts this disease?
Recall    = Did the model find most real examples of this disease?
F1-score  = Balance between trustworthiness and finding ability
Support   = Number of real examples for that class
```

F1-score becomes high only when both precision and recall are strong.

If either precision or recall is very low, the F1-score also drops.

---

## 8. Plant Disease Interpretation

For plant disease detection:

```text
Low precision = many false alarms
Low recall    = many missed diseases
Low F1-score  = poor balance between false alarms and missed diseases
```

If a disease is dangerous, recall may be especially important because missing the disease could cause real damage.

If false alarms are costly, precision becomes more important.

A strong model should ideally have both high precision and high recall.

### Support

Support means the number of real validation examples for that class.

## Ask Students

### Q1. What does precision tell us?

**Answer:**
Whether we can trust the model when it predicts a certain class.

### Q2. What does recall tell us?

**Answer:**
Whether the model finds most real examples of a class or misses many.

### Q3. What does support mean?

**Answer:**
How many actual validation images belong to that class.

### Q4. Why is F1-score useful?

**Answer:**
It combines precision and recall into one balanced score.

---

# Cell 14 — Check Output Files

```python
files_to_check = [
    "part3_confusion_matrix.png",
    "part3_class_accuracies.png",
    "part3_predictions_random.png",
    "part3_predictions_wrong.png"
]

for f in files_to_check:
    print(f, "exists:", os.path.exists(f))
```

## Explain After Cell 14

This checks whether all result images were saved.

These images can be used later in slides, reports, or final student presentations.

## Ask Students

### Q1. Why do we save the output images?

**Answer:**
So we can reuse them later for analysis, reports, or presentations.

---

# Part 3 Closing Summary

Use this at the end:

> In Part 3, we evaluated the model trained in Part 2. We learned that accuracy alone is not enough. We used a confusion matrix to see which classes the model confuses. We looked at per-class accuracy to find the easiest and hardest diseases. We inspected correct and wrong predictions visually. Finally, we used precision, recall, F1-score, and support to better understand model performance.

---

# Final Discussion Questions

## Q1. Why is validation accuracy more useful than training accuracy?

**Answer:**
Validation accuracy tells us how well the model performs on images it did not directly learn from.

## Q2. What does the confusion matrix show that accuracy does not?

**Answer:**
It shows exactly which classes are being confused.

## Q3. What is the difference between precision and recall?

**Answer:**
Precision asks whether the model’s predictions are trustworthy. Recall asks whether the model catches most real examples of a class.

## Q4. Why should we inspect wrong predictions visually?

**Answer:**
Because visual inspection helps us understand whether mistakes are caused by similar-looking diseases, unclear images, or model weakness.

## Q5. What could we do to improve the model after seeing its mistakes?

**Answer:**
Collect more data, use data augmentation, improve image quality, train longer, or tune model settings.
