import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
from transformers import CLIPModel

def load_pretrained_model(model_name):
    """
    Load a pretrained model and remove its final classification layer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        feature_dim = 512
        transform = Compose([
            Grayscale(num_output_channels=3),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.fc = nn.Identity()
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(pretrained=True)
        feature_dim = 1280
        transform = Compose([
            Grayscale(num_output_channels=3),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.classifier = nn.Identity()
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        feature_dim = 1280
        transform = Compose([
            Grayscale(num_output_channels=3),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.classifier = nn.Identity()
    elif model_name == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        feature_dim = 1024
        transform = Compose([
            Grayscale(num_output_channels=3),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.fc = nn.Identity()
    elif model_name == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
        feature_dim = 512
        transform = Compose([
            Grayscale(num_output_channels=3),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    elif model_name == "convnext_tiny":
        model = create_model("convnext_tiny", pretrained=True)
        feature_dim = 768
        transform = Compose([
            Grayscale(num_output_channels=3),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()
    return model, feature_dim, transform

def extract_features(model, dataloader, feature_dim, model_name):
    """
    Extracts features using a given model and DataLoader.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, labels_list, image_paths = [], [], []

    for images, labels, paths in dataloader:
        images = images.to(device)
        with torch.no_grad():
            features = model(images)
            if model_name in ["resnet18", "efficientnet_b0"]:
              features = model(images)
            elif model_name == "shufflenet_v2":
              features = features.view(features.size(0), -1)  # Flatten for ShuffleNetV2

            features = features / features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu().numpy())
        labels_list.extend(labels.numpy())
        image_paths.extend(paths)
    return np.vstack(embeddings), np.array(labels_list), image_paths
