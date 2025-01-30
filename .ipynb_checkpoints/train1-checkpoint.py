from datasets import load_dataset
from model_utils import load_pretrained_model, extract_features
from dataset_utils import FashionProductDataset
from faiss_utils import build_faiss_index, search_faiss
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from plot_utils import plot_nearest_neighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import faiss
import pickle
import numpy as np
model_name = "mobilenet_v3"  # Change to "resnet18", "effientnet_b0", "mobilenet_v3", "shufflenet_v2", etc.

# Data prepration
# Load the dataset
dataset = load_dataset("ashraq/fashion-product-images-small")

# Define label column
label_column = "subCategory"

# Extract images and labels
images = [sample["image"] for sample in dataset["train"]]
labels = [sample[label_column] for sample in dataset["train"]]

# Create a DataFrame for labels
label_df = pd.DataFrame({'label': labels})

# Count the number of samples per class
class_counts = label_df['label'].value_counts()

# Define label column
label_column = "subCategory"

# Extract images and labels
images = [sample["image"] for sample in dataset["train"]]
labels = [sample[label_column] for sample in dataset["train"]]

# Create a DataFrame for labels
label_df = pd.DataFrame({'label': labels})

# Count the number of samples per class
class_counts = label_df['label'].value_counts()

# Get the class names for the top 10 classes
top_10_classes = class_counts.head(10)
top_10_class_names = top_10_classes.index
# Filter images and labels together to ensure alignment
filtered_data = [(img, lbl) for img, lbl in zip(images, labels) if lbl in top_10_class_names]

# Unzip the filtered data
filtered_images, filtered_labels = zip(*filtered_data)

# Convert to lists
filtered_images = list(filtered_images)
filtered_labels = list(filtered_labels)

# Encode only the top 15 classes
label_encoder = LabelEncoder()
filtered_labels = label_encoder.fit_transform(filtered_labels)

# Stratified splitting
train_images, temp_images, train_labels, temp_labels = train_test_split(
    filtered_images, filtered_labels, test_size=0.3, random_state=42, stratify=filtered_labels
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train size: {len(train_images)}, Validation size: {len(val_images)}, Test size: {len(test_images)}")
print(f"Train size: {len(train_labels)}, Validation size: {len(val_labels)}, Test size: {len(test_labels)}")

import os
# Create directories for train, validation, and test images
os.makedirs("train_images", exist_ok=True)
os.makedirs("val_images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

# Function to save images
def save_images(image_list, folder, prefix):
    image_paths = []
    for i, img in enumerate(image_list):
        image_path = os.path.join(folder, f"{prefix}_{i}.jpg")
        img.save(image_path)  # Save PIL image
        image_paths.append(image_path)  # Store path for later use
    return image_paths

# Save images and get their file paths
train_image_paths = save_images(train_images, "train_images", "train")
val_image_paths = save_images(val_images, "val_images", "val")
test_image_paths = save_images(test_images, "test_images", "test")

import pickle

# Save labels as a dictionary
label_data = {
    "train": (train_image_paths, train_labels),
    "val": (val_image_paths, val_labels),
    "test": (test_image_paths, test_labels),
}

# Save to disk
with open("image_labels.pkl", "wb") as f:
    pickle.dump(label_data, f)
    
# Obtain embedding form pre-trained model 
# Load the pre-trained model and transformation
model, feature_dim, transform = load_pretrained_model(model_name)
# Load the stored image paths and labels
with open("image_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

train_image_paths, train_labels = label_data["train"]

# Create dataset instances
train_dataset = FashionProductDataset(train_image_paths, train_labels, transform=transform)


print(f"Train dataset: {len(train_dataset)} images")

import faiss
import pickle
import numpy as np

# Extract features
train_embeddings, train_labels, train_image_paths = extract_features(model, train_loader, feature_dim, model_name)

# Create FAISS index
index = build_faiss_index(train_embeddings)

# Save FAISS index
faiss.write_index(index, f"faiss_index_{model_name}.bin")

# Save image paths
with open("train_image_paths.pkl", "wb") as f:
    pickle.dump(train_image_paths, f)

# Validation
# Load the stored image paths and labels
with open("image_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

train_image_paths, train_labels = label_data["train"]

# Create dataset instances
train_dataset = FashionProductDataset(train_image_paths, train_labels, transform=transform)


print(f"Train dataset: {len(train_dataset)} images")
    
# Load the stored image paths and labels
with open("image_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

val_image_paths, val_labels = label_data["val"]

# Create dataset instances
val_dataset = FashionProductDataset(val_image_paths, val_labels, transform=transform)
print(f"Val dataset: {len(val_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Extract features
valid_embeddings, valid_labels, valid_image_paths = extract_features(model, val_loader, feature_dim, model_name)

# Load FAISS index and image paths
index = faiss.read_index(f"faiss_index_{model_name}.bin")

with open("train_image_paths.pkl", "rb") as f:
    train_image_paths = pickle.load(f)

from collections import defaultdict
def compute_metrics(indices, subset_labels, train_labels, k=5):
    """
    Computes Precision@K, Recall@K, Retrieval Accuracy, Per-Class Precision, Weighted Precision, and Mean Reciprocal Rank.
    """
    correct_retrievals = 0
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []
    per_class_precision = defaultdict(list)
    class_counts = defaultdict(int)

    for label in train_labels:
        class_counts[label] += 1

    for query_idx, query_label in enumerate(subset_labels):
        retrieved_labels = [train_labels[idx] for idx in indices[query_idx]]
        relevant_retrieved = sum(1 for lbl in retrieved_labels if lbl == query_label)

        precision_at_k = relevant_retrieved / k
        precision_scores.append(precision_at_k)
        per_class_precision[query_label].append(precision_at_k)

        recall_at_k = relevant_retrieved / class_counts[query_label]
        recall_scores.append(recall_at_k)

        try:
            rank = retrieved_labels.index(query_label) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)

        if retrieved_labels[0] == query_label:
            correct_retrievals += 1

    retrieval_accuracy = correct_retrievals / len(subset_labels)
    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)

    weighted_precision = sum(
        (sum(per_class_precision[label]) / len(per_class_precision[label])) * (class_counts[label] / len(train_labels))
        for label in per_class_precision
    )

    return mean_precision, mean_recall, retrieval_accuracy, weighted_precision, mean_reciprocal_rank, per_class_precision


# Search FAISS for nearest neighbors
k = 5
distances, indices = index.search(valid_embeddings, k)

precision, recall, retrieval_acc, weighted_precision, mrr, per_class_prec = compute_metrics(indices, valid_labels, train_labels, k)
print(f"Precision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")
print(f"Retrieval Accuracy: {retrieval_acc:.4f}")
print(f"Weighted Precision@{k}: {weighted_precision:.4f}")
print(f"Mean Reciprocal Rank: {mrr:.4f}")

# Print per-class precision
for class_label, prec_values in per_class_prec.items():
    avg_prec = sum(prec_values) / len(prec_values)
    print(f"Class {class_label} Precision@{k}: {avg_prec:.4f}")