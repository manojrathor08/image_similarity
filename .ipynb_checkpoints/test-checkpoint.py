import pickle
import faiss
import torch
from torch.utils.data import DataLoader
from dataset_utils import FashionProductDataset
from model_utils import load_pretrained_model, extract_features
from collections import defaultdict
from faiss_utils import search_faiss

# Load FAISS index
model_name = "mobilenet_v3"
index = faiss.read_index(f"faiss_index_{model_name}.bin")

# Load validation dataset
with open("image_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

val_paths, val_labels = label_data["val"]
train_paths, train_labels = label_data["train"]

# Load model
model, feature_dim, transform = load_pretrained_model(model_name)
val_dataset = FashionProductDataset(val_paths, val_labels, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Extract features
val_embeddings, val_labels, val_paths = extract_features(model, val_loader, feature_dim, model_name)

# Perform FAISS search
k = 5
distances, indices = index.search(val_embeddings, k)

# Compute Metrics
def compute_metrics(indices, val_labels, train_labels, k=5):
    correct_retrievals = 0
    precision_scores, recall_scores, reciprocal_ranks = [], [], []
    per_class_precision, class_counts = defaultdict(list), defaultdict(int)

    for label in train_labels:
        class_counts[label] += 1

    for query_idx, query_label in enumerate(val_labels):
        retrieved_labels = [train_labels[idx] for idx in indices[query_idx]]
        relevant_retrieved = sum(1 for lbl in retrieved_labels if lbl == query_label)

        precision_scores.append(relevant_retrieved / k)
        per_class_precision[query_label].append(relevant_retrieved / k)
        recall_scores.append(relevant_retrieved / class_counts[query_label])

        try:
            rank = retrieved_labels.index(query_label) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)

        if retrieved_labels[0] == query_label:
            correct_retrievals += 1

    retrieval_accuracy = correct_retrievals / len(val_labels)
    weighted_precision = sum(
        (sum(per_class_precision[label]) / len(per_class_precision[label])) * (class_counts[label] / len(train_labels))
        for label in per_class_precision
    )

    return (
        sum(precision_scores) / len(precision_scores),
        sum(recall_scores) / len(recall_scores),
        retrieval_accuracy,
        weighted_precision,
        sum(reciprocal_ranks) / len(reciprocal_ranks),
    )

precision, recall, retrieval_acc, weighted_precision, mrr = compute_metrics(indices, val_labels, train_labels, k)

print(f"Precision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")
print(f"Retrieval Accuracy: {retrieval_acc:.4f}")
print(f"Weighted Precision@{k}: {weighted_precision:.4f}")
print(f"Mean Reciprocal Rank: {mrr:.4f}")

# Print per-class precision
for class_label, prec_values in per_class_prec.items():
    avg_prec = sum(prec_values) / len(prec_values)
    print(f"Class {class_label} Precision@{k}: {avg_prec:.4f}")