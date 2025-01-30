from torch.utils.data import Dataset
from PIL import Image

class FashionProductDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of file paths to images.
            labels: Corresponding labels.
            transform: PyTorch transformation pipeline.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")  # Load from disk

        if self.transform:
            transformed_image = self.transform(image)  # Apply transformation
            return transformed_image, label, image_path  # Return file path instead of image object

        return image, label, image_path
