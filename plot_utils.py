import matplotlib.pyplot as plt
from PIL import Image

# Define the class mapping 
class_mapping = {
    0: 'Bags', 
    1: 'Bottomwear', 
    2: 'Eyewear', 
    3: 'Fragrance', 
    4: 'Innerwear', 
    5: 'Jewellery', 
    6: 'Sandal', 
    7: 'Shoes', 
    8: 'Topwear', 
    9: 'Watches'
}

def plot_nearest_neighbors(query_index, query_image, nearest_indices, train_image_paths, train_labels, subset_labels, k):
    """Plot the query image and its k nearest neighbors."""
    fig, axes = plt.subplots(1, k + 1, figsize=(15, 5))
    
    # Plot query image
    query_label = subset_labels[query_index]
    query_class = class_mapping.get(query_label, "Unknown")  # Map label to class
    axes[0].imshow(query_image)
    axes[0].set_title(f"Query\nLabel: {query_label} ({query_class})")
    axes[0].axis("off")

    # Plot nearest neighbors
    for rank, neighbor_idx in enumerate(nearest_indices):
        neighbor_image_path = train_image_paths[neighbor_idx]  # Get image path
        neighbor_label = train_labels[neighbor_idx]  # Get label
        neighbor_class = class_mapping.get(neighbor_label, "Unknown")  # Map label to class
        
        neighbor_image = Image.open(neighbor_image_path).convert("RGB")  # Load image from disk
        axes[rank + 1].imshow(neighbor_image)
        axes[rank + 1].set_title(f"Neighbor {rank + 1}\nLabel: {neighbor_label} ({neighbor_class})")
        axes[rank + 1].axis("off")
    
    plt.tight_layout()
    plt.show()
