import pickle
import torch
import faiss
from torch.utils.data import DataLoader
from dataset_utils import FashionProductDataset
from model_utils import load_pretrained_model, extract_features
from faiss_utils import build_faiss_index

# Load dataset
model_name = "mobilenet_v3"
with open("image_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

train_paths, train_labels = label_data["train"]

# Load model
model, feature_dim, transform = load_pretrained_model(model_name)
train_dataset = FashionProductDataset(train_paths, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Extract features
train_embeddings, train_labels, train_paths = extract_features(model, train_loader, feature_dim, model_name)

# Build FAISS index
index = build_faiss_index(train_embeddings)
faiss.write_index(index, f"faiss_index_{model_name}.bin")

# Save image paths
with open("train_image_paths.pkl", "wb") as f:
    pickle.dump(train_paths, f)

print("Training complete! FAISS index and embeddings saved.")
