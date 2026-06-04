# Plant Disease Detection with AI — Part 1 Instructor Monitor

## Part 1: Meet the Dataset

In Part 1, students explore the plant disease dataset before training any model.

The original Part 1 material focuses on understanding Machine Learning, loading the PlantVillage-style dataset, counting classes, displaying leaf images, and inspecting image pixels/RGB channels.

---

# Big Picture Explanation

Use this before starting the notebook:

> Today we are not training AI yet. We are meeting the data. Before an AI model can learn, we need to understand what examples we are giving it.

Simple flow:

```text
Load dataset
      ↓
List disease classes
      ↓
Count images per class
      ↓
Show healthy and diseased leaves
      ↓
Look at image shape and pixel values
      ↓
Understand that images are numbers
```

Main idea:

> A computer does not see a leaf the way humans do. It sees a grid of numbers.

---

# Cell 1 — Import Libraries

```python
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
```

## Explain After Cell 1

This cell imports the tools we need.

Important libraries:

* `os`: helps us work with folders and file paths.
* `random`: helps us pick random images.
* `matplotlib.pyplot`: helps us draw charts and display images.
* `matplotlib.image`: helps us read image files.
* `numpy`: helps us work with numerical data.

## Ask Students

### Q1. Why do we need Python libraries?

**Answer:**
Libraries give us ready-made tools so we do not have to build everything from scratch.

### Q2. Why do we need `os`?

**Answer:**
Because our dataset is stored in folders, and `os` helps Python find and read those folders.

### Q3. Why do we use `matplotlib`?

**Answer:**
To show images and create charts.

---

# Cell 2 — Set Dataset Path

```python
DATASET_PATH = "/fs/ess/PAS2699/osc_summer_camp_2026/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")

print("Train folder exists:", os.path.exists(TRAIN_DIR))
print("Valid folder exists:", os.path.exists(VALID_DIR))

print("Train path:", TRAIN_DIR)
print("Valid path:", VALID_DIR)
```

Expected output:

```text
Train folder exists: True
Valid folder exists: True
```

## Explain After Cell 2

The dataset has two main folders:

```text
train/
valid/
```

Training images are used later for model learning.

Validation images are used later to check how well the model performs.

In Part 1, we are only exploring the dataset.

## Ask Students

### Q1. What does `DATASET_PATH` mean?

**Answer:**
It tells Python where the dataset is stored.

### Q2. Why do we check whether the folders exist?

**Answer:**
To make sure Python can actually find the data before we continue.

### Q3. What does `True` mean here?

**Answer:**
It means the folder exists and Python found it successfully.

---

# Cell 3 — List All Classes

```python
all_classes = sorted(os.listdir(TRAIN_DIR))

print("Total number of classes:", len(all_classes))
print()

for i, cls in enumerate(all_classes):
    print(f"{i+1:>2}. {cls}")
```

## Explain After Cell 3

Each folder inside `train/` is one class.

A **class** means a category.

Example classes:

```text
Apple___healthy
Apple___Black_rot
Tomato___healthy
Tomato___Early_blight
Potato___healthy
```

The folder name is the label for all images inside that folder.

## Ask Students

### Q1. What is a class?

**Answer:**
A class is a category that an image can belong to.

### Q2. What is a label?

**Answer:**
A label is the correct answer for an image.

### Q3. Where do the labels come from in this dataset?

**Answer:**
They come from the folder names.

### Q4. If an image is inside `Tomato___healthy`, what is its label?

**Answer:**
`Tomato___healthy`.

---

# Cell 4 — Count Images Per Class

```python
class_counts = {}

for cls in all_classes:
    class_folder = os.path.join(TRAIN_DIR, cls)
    num_images = len(os.listdir(class_folder))
    class_counts[cls] = num_images

total_train = sum(class_counts.values())

print("Total training images:", total_train)
print()

top10 = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 classes by image count:")
for cls, count in top10:
    print(f"{cls:<45} {count:>5} images")
```

## Explain After Cell 4

This cell counts how many images are in each class.

This is important because before training AI, we need to understand the dataset.

We want to know:

```text
How many images do we have?
Are all classes similar in size?
Are some classes much bigger than others?
```

## Ask Students

### Q1. Why do we count images in each class?

**Answer:**
To understand how much data we have for each category.

### Q2. What could happen if one class has many images and another class has very few?

**Answer:**
The model may learn the larger class better and perform poorly on the smaller class.

### Q3. What is class imbalance?

**Answer:**
Class imbalance means some classes have many more examples than others.

---

# Cell 5 — Plot Class Distribution

```python
short_labels = [c.replace("_", "\n") for c in class_counts.keys()]
counts = list(class_counts.values())

plt.figure(figsize=(20, 6))
plt.bar(range(len(counts)), counts)
plt.xticks(range(len(short_labels)), short_labels, fontsize=6, rotation=45, ha="right")
plt.ylabel("Number of Images")
plt.title("Number of Training Images per Disease Class")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
```

## Explain After Cell 5

This cell creates a bar chart showing how many images each class has.

The chart makes it easier to see class imbalance.

It also saves the chart as:

```text
class_distribution.png
```

## Ask Students

### Q1. Why is a chart useful here?

**Answer:**
A chart helps us quickly see which classes have more or fewer images.

### Q2. What does a tall bar mean?

**Answer:**
That class has many images.

### Q3. What does a short bar mean?

**Answer:**
That class has fewer images.

### Q4. Why might class imbalance matter later?

**Answer:**
The model may become better at classes it sees more often and worse at classes it sees less often.

---

# Cell 6 — Define Function to Show Random Images

```python
def show_random_images(class_name, num_images=4):
    folder = os.path.join(TRAIN_DIR, class_name)

    if not os.path.exists(folder):
        print("Class folder does not exist:", folder)
        return

    all_files = os.listdir(folder)
    chosen = random.sample(all_files, min(num_images, len(all_files)))

    fig, axes = plt.subplots(1, len(chosen), figsize=(14, 4))
    fig.suptitle(f"Class: {class_name}", fontsize=13, fontweight="bold")

    if len(chosen) == 1:
        axes = [axes]

    for ax, filename in zip(axes, chosen):
        img_path = os.path.join(folder, filename)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{img.shape[1]}x{img.shape[0]}", fontsize=9)

    plt.tight_layout()
    plt.show()
```

## Explain After Cell 6

This cell creates a reusable function.

A function is a block of code we can call again and again.

This function:

```text
takes a class name
chooses random images from that class
displays those images
```

## Ask Students

### Q1. What is a function?

**Answer:**
A function is reusable code that performs a specific task.

### Q2. What does this function need as input?

**Answer:**
It needs the class name and optionally the number of images to show.

### Q3. Why do we use random images?

**Answer:**
So we can see different examples from the same class each time.

---

# Cell 7 — Display Healthy and Diseased Tomato Leaves

```python
show_random_images("Tomato___healthy")
show_random_images("Tomato___Early_blight")
```

## Explain After Cell 7

Now we use the function to show images.

We compare:

```text
healthy tomato leaves
diseased tomato leaves
```

Students should observe visual differences.

Possible clues:

```text
spots
yellowing
brown areas
dry edges
color changes
damage patterns
```

## Ask Students

### Q1. What do you notice about the healthy leaves?

**Answer:**
They may look greener, cleaner, and more uniform.

### Q2. What do you notice about the diseased leaves?

**Answer:**
They may have spots, discoloration, brown patches, or damaged areas.

### Q3. What clues might a model learn from these images?

**Answer:**
It may learn patterns related to color, texture, spots, shapes, and damage.

---

# Cell 8 — Define Function to Compare Healthy vs Diseased Leaves

```python
def compare_healthy_vs_diseased(healthy_class, diseased_class):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Healthy vs Diseased — Can you spot the difference?", fontsize=13)

    for ax, cls, label in zip(
        axes,
        [healthy_class, diseased_class],
        ["Healthy", "Diseased"]
    ):
        folder = os.path.join(TRAIN_DIR, cls)

        if not os.path.exists(folder):
            print("Missing class:", cls)
            continue

        filename = random.choice(os.listdir(folder))
        img = mpimg.imread(os.path.join(folder, filename))

        ax.imshow(img)
        ax.set_title(label + "\n" + cls, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
```

Run:

```python
compare_healthy_vs_diseased(
    healthy_class="Tomato___healthy",
    diseased_class="Tomato___Early_blight"
)
```

Try another example:

```python
compare_healthy_vs_diseased(
    healthy_class="Potato___healthy",
    diseased_class="Potato___Early_blight"
)
```

If a class name does not work, search for the correct class name:

```python
[c for c in all_classes if "Potato" in c]
```

## Explain After Cell 8

This function puts one healthy and one diseased image side by side.

This helps students visually compare classes.

This is also close to what image classification does:

```text
look at image → decide which category it belongs to
```

## Ask Students

### Q1. Why is side-by-side comparison useful?

**Answer:**
It helps us clearly see differences between healthy and diseased leaves.

### Q2. Could some diseases be hard even for humans to tell apart?

**Answer:**
Yes. Some diseases may look similar, especially in early stages or poor lighting.

### Q3. If humans find some images difficult, what does that mean for AI?

**Answer:**
The model may also struggle with those cases.

---

# Cell 9 — Inspect Image Shape and Pixel Values

```python
sample_folder = os.path.join(TRAIN_DIR, "Tomato___healthy")
sample_file = os.listdir(sample_folder)[0]
sample_img = mpimg.imread(os.path.join(sample_folder, sample_file))

print("Image shape:", sample_img.shape)
print("Height:", sample_img.shape[0])
print("Width:", sample_img.shape[1])
print("Color channels:", sample_img.shape[2])

print("\nTop-left 4x4 pixel values from Red channel:")
print(sample_img[:4, :4, 0])
```

## Explain After Cell 9

This cell shows how the computer stores the image.

If image shape is:

```text
(256, 256, 3)
```

that means:

```text
256 = height
256 = width
3 = color channels: Red, Green, Blue
```

This line:

```python
sample_img[:4, :4, 0]
```

means:

```text
first 4 rows
first 4 columns
red channel only
```

So it prints the red values from the top-left 4 by 4 pixels.

## Ask Students

### Q1. What is a pixel?

**Answer:**
A pixel is one tiny square of color in an image.

### Q2. What does image shape mean?

**Answer:**
It tells us the height, width, and number of color channels in the image.

### Q3. What are the 3 color channels?

**Answer:**
Red, Green, and Blue.

### Q4. What does `sample_img[:4, :4, 0]` show?

**Answer:**
It shows the red-channel values from the top-left 4 by 4 pixels.

### Q5. Why do we look at pixel values?

**Answer:**
To understand that computers see images as numbers.

---

# Cell 10 — Show RGB Color Channels

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("What a Computer Sees — RGB Channels", fontsize=13)

axes[0].imshow(sample_img)
axes[0].set_title("Original")
axes[0].axis("off")

channel_names = ["Red channel", "Green channel", "Blue channel"]
cmaps = ["Reds", "Greens", "Blues"]

for i in range(3):
    axes[i+1].imshow(sample_img[:, :, i], cmap=cmaps[i])
    axes[i+1].set_title(channel_names[i])
    axes[i+1].axis("off")

plt.tight_layout()
plt.show()
```

## Explain After Cell 10

A color image is made from three layers:

```text
Red
Green
Blue
```

These are called channels.

Humans see the full image, but the computer sees RGB number layers.

For plant disease detection:

```text
healthy leaves may have strong green patterns
diseased leaves may show yellow, brown, or dark patterns
```

## Ask Students

### Q1. What is a color channel?

**Answer:**
A color channel is one layer of color information, such as red, green, or blue.

### Q2. Why do we separate the RGB channels?

**Answer:**
To see how the image is built from separate color-number layers.

### Q3. Why are color patterns useful for plant disease detection?

**Answer:**
Diseased leaves may have different colors, such as yellowing, brown spots, or dark patches.

---

# Cell 11 — Find Smallest and Largest Classes

```python
smallest_class = min(class_counts.items(), key=lambda x: x[1])
largest_class = max(class_counts.items(), key=lambda x: x[1])

print("Smallest class:", smallest_class[0], "-", smallest_class[1], "images")
print("Largest class:", largest_class[0], "-", largest_class[1], "images")
```

## Explain After Cell 11

This cell finds:

```text
class with the fewest images
class with the most images
```

This helps identify class imbalance.

Class imbalance means some classes have many more images than others.

## Ask Students

### Q1. What does the smallest class tell us?

**Answer:**
It tells us which class has the fewest examples.

### Q2. What does the largest class tell us?

**Answer:**
It tells us which class has the most examples.

### Q3. Why can imbalance be a problem?

**Answer:**
The model may learn classes with many images better than classes with few images.

### Q4. How could we fix class imbalance?

**Answer:**
Collect more data, use data augmentation, oversample small classes, or use class weights during training.

---

# Cell 12 — Search Classes by Plant Name

```python
plant_name = "Apple"

matching_classes = [cls for cls in all_classes if plant_name in cls]

print(f"Classes containing '{plant_name}':")
for cls in matching_classes:
    print(cls)
```

Try changing the plant name:

```python
plant_name = "Potato"
```

```python
plant_name = "Corn"
```

```python
plant_name = "Grape"
```

## Explain After Cell 12

This helps students search the dataset.

Instead of reading all class names manually, they can filter classes by plant name.

## Ask Students

### Q1. Why is searching class names useful?

**Answer:**
Because the dataset has many classes, and searching helps quickly find classes for one plant.

### Q2. What plant did you search for?

**Answer:**
Students can answer based on their chosen plant.

### Q3. How many disease classes did that plant have?

**Answer:**
Students count the matching results.

---

# Cell 13 — Try Your Own Example

```python
show_random_images("Apple___healthy")
show_random_images("Apple___Black_rot")
```

If the class name does not work, search for the correct class name:

```python
[c for c in all_classes if "Apple" in c]
```

Students should answer:

```text
1. Which plant did you choose?
2. Which healthy class did you display?
3. Which diseased class did you display?
4. What visual difference did you observe?
5. Which class has the fewest images?
```

## Explain After Cell 13

This is the hands-on exploration activity.

Students choose their own plant and disease class.

The goal is to practice:

```text
searching class names
displaying images
observing visual patterns
connecting images to labels
```

## Ask Students

### Q1. Which plant did you choose?

**Answer:**
Student-specific.

### Q2. What difference did you observe between healthy and diseased images?

**Answer:**
Possible answers: spots, color changes, damaged areas, yellowing, brown patches.

### Q3. Could the model use similar clues?

**Answer:**
Yes. The model may learn numerical patterns related to these visual differences.

---

# Cell 14 — Part 1 Wrap-Up

```python
print("Part 1 Complete")
print()
print("Today we learned:")
print("- AI can learn patterns from examples.")
print("- Image classification means assigning a label to an image.")
print("- A dataset contains images and labels.")
print("- Each folder name is a class label.")
print("- Images are stored as pixel values.")
print("- Color images have Red, Green, and Blue channels.")
print()
print("Next: Part 2 will prepare images for model training.")
```

## Explain After Cell 14

Summarize the main lesson:

> Today we did not train AI yet. We explored the data. We learned that image folders become labels, images are made of pixels, and pixels contain RGB numbers.

## Ask Students

### Q1. What is the most important thing we learned today?

**Answer:**
That images are data, and the model learns from labeled examples.

### Q2. Why do we explore data before training?

**Answer:**
To understand what examples the model will learn from and to check for issues like class imbalance.

---

# Part 1 Closing Summary

Use this at the end:

> In Part 1, we explored the plant disease dataset. We listed all the classes, counted images per class, displayed healthy and diseased leaves, and inspected image shape and RGB pixel values. The key idea is that humans see leaf images, but computers see numerical data. In the next part, we will use this data to train an AI model.

---

# Final Discussion Questions

## Q1. What is a dataset?

**Answer:**
A collection of examples used to train or evaluate a model.

## Q2. What is a label?

**Answer:**
The correct answer or category for an image.

## Q3. What is image classification?

**Answer:**
Assigning an image to one category or class.

## Q4. What is a pixel?

**Answer:**
One tiny square of color in an image.

## Q5. What are RGB channels?

**Answer:**
The red, green, and blue color layers that make up a color image.

## Q6. What does class imbalance mean?

**Answer:**
Some classes have many more images than others.

## Q7. Why does class imbalance matter?

**Answer:**
The model may learn larger classes better and smaller classes worse.

## Q8. Why do we look at pixel values?

**Answer:**
To understand that computers process images as numbers.

## Q9. What visual clues can suggest a leaf is diseased?

**Answer:**
Spots, yellowing, brown areas, dry edges, holes, or unusual color patterns.

## Q10. What will we do next?

**Answer:**
Train a model using the dataset.
