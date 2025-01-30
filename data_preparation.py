import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
dataset = load_dataset("ashraq/fashion-product-images-small")
label_column = "subCategory"

# Extract images & labels
images = [sample["image"] for sample in dataset["train"]]
labels = [sample[label_column] for sample in dataset["train"]]

# Find top 10 categories
label_df = pd.DataFrame({'label': labels})
top_10_classes = label_df['label'].value_counts().head(10).index
filtered_data = [(img, lbl) for img, lbl in zip(images, labels) if lbl in top_10_classes]

# Split data
filtered_images, filtered_labels = zip(*filtered_data)
filtered_labels = LabelEncoder().fit_transform(filtered_labels)

train_images, temp_images, train_labels, temp_labels = train_test_split(
    filtered_images, filtered_labels, test_size=0.3, random_state=42, stratify=filtered_labels
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train size: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}")

# Save images
os.makedirs("train_images", exist_ok=True)
os.makedirs("val_images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

def save_images(image_list, folder, prefix):
    paths = []
    for i, img in enumerate(image_list):
        path = os.path.join(folder, f"{prefix}_{i}.jpg")
        img.save(path)
        paths.append(path)
    return paths

train_paths = save_images(train_images, "train_images", "train")
val_paths = save_images(val_images, "val_images", "val")
test_paths = save_images(test_images, "test_images", "test")

# Save label mappings
label_data = {
    "train": (train_paths, train_labels),
    "val": (val_paths, val_labels),
    "test": (test_paths, test_labels),
}

with open("image_labels.pkl", "wb") as f:
    pickle.dump(label_data, f)

print("Dataset preparation complete! Saved images and labels.")
