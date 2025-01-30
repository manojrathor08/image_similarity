# 🔍 Image Similarity Search using Deep Learning

This project implements a **content-based image retrieval (CBIR) system** using deep learning models to find visually similar images. The system extracts feature embeddings from pre-trained models and performs **FAISS-based similarity search**.

---

## 📂 Dataset Information

We use the **Fashion Product Images Small** dataset from Hugging Face 🤗:

- **Dataset Source:** [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- **Total Images:** **44,072**
- **Total Classes:** **45**
- **Selected Classes:** **Top 10 most frequent classes**
- **Data Split:** Stratified split into **Training, Validation, and Test sets**.

### **📊 Data Splitting Strategy**
The dataset was divided in a **stratified manner** to maintain class balance:
- **Train Set:** **70%** of the selected data
- **Validation Set:** **15%** (used for evaluation)
- **Test Set:** **15%** (used for final model testing)

---

## 🚀 **Project Workflow**

### 1️⃣ **Data Preprocessing**
- Filter top **10 classes** from the original **45 classes**.
- Load images and labels, then split data into **train, validation, and test** sets.

### 2️⃣ **Feature Extraction**
- Used **pre-trained deep learning models** to extract embeddings.
- Models Evaluated: **ResNet-18, EfficientNet-B0, MobileNetV3, SqueezeNet, ShuffleNet-V2, ConvNeXt-Tiny**.

### 3️⃣ **Similarity Search with FAISS**
- Built a **FAISS index** to store and retrieve image embeddings.
- Indexed training images and used FAISS to find the **5 most similar images** for any query.

### 4️⃣ **Model Evaluation**
- Computed **Precision@5, Retrieval Accuracy, Weighted Precision, and Mean Reciprocal Rank (MRR)**.
- Compared different models to select the best balance of **accuracy and speed**.

### 5️⃣ **UI Deployment on Hugging Face Spaces**
- Built an **interactive demo using Gradio**.
- Allows users to **upload an image and retrieve 5 similar images** from the dataset.

---

## 📊 **Model Performance Comparison**

| Model              | Precision@5 | Retrieval Accuracy | Weighted Precision@5 | Mean Reciprocal Rank |
|--------------------|------------|---------------------|----------------------|----------------------|
| **ResNet-18**     | 0.9651      | 0.9778             | 0.9651               | 0.9848               |
| **ShuffleNet-V2**  | 0.9666      | 0.9796             | 0.9666               | 0.9862               |
| **EfficientNet-B0** | 0.9695     | 0.9794             | 0.9695               | 0.9865               |
| **MobileNetV3**    | 0.9701      | 0.9805             | 0.9700               | 0.9867               |
| **SqueezeNet**     | 0.9610      | 0.9767             | 0.9610               | 0.9837               |
| **ConvNeXt-Tiny**  | 0.9739      | 0.9816             | 0.9739               | 0.9873               |

---

## 🔬 **Per-Class Precision Analysis**

To better understand how the model performs on individual classes, we computed **Precision@5 for each class**.

| Class Label | ResNet-18 | SqueezeNet | EfficientNet-B0 | MobileNet-V3 | ConvNeXt-Tiny | ShuffleNet-V2 |
|------------|----------|------------|----------------|--------------|--------------|---------------|
| **0**      | 0.9699   | 0.9620     | 0.9699         | 0.9729       | 0.9834       | 0.9672        |
| **1**      | 0.9692   | 0.9658     | 0.9677         | 0.9727       | 0.9777       | 0.9722        |
| **2**      | 1.0000   | 1.0000     | 1.0000         | 1.0000       | 1.0000       | 1.0000        |
| **3**      | 0.9413   | 0.9413     | 0.9680         | 0.9667       | 0.9573       | 0.9333        |
| **4**      | 0.8827   | 0.8731     | 0.9188         | 0.9063       | 0.9262       | 0.8952        |
| **5**      | 0.9068   | 0.8975     | 0.9605         | 0.9543       | 0.9753       | 0.9395        |
| **6**      | 0.7125   | 0.6861     | 0.7611         | 0.7306       | 0.8069       | 0.7125        |
| **7**      | 0.9699   | 0.9617     | 0.9612         | 0.9679       | 0.9685       | 0.9672        |
| **8**      | 0.9876   | 0.9873     | 0.9889         | 0.9891       | 0.9881       | 0.9883        |
| **9**      | 0.9780   | 0.9780     | 0.9843         | 0.9859       | 0.9796       | 0.9843        |

### **Observations from Per-Class Precision**
- Class **2** achieved the **highest precision (1.0000)**, meaning all top-5 retrieved images were correct.
- Class **6** had the lowest precision (0.7111), indicating that retrieval accuracy for this class can be improved.
- Most classes have high precision, showing that the model retrieves relevant images effectively.

---

## 🏆 **Final Model Selection for UI Deployment**
Based on **accuracy, latency, and model size**, the best models for deployment are:

1️⃣ **Swin Transformer** 🏆 (Highest retrieval accuracy & MRR)  
2️⃣ **EfficientNet-B0** 🚀 (Best balance of speed & accuracy)  
3️⃣ **ConvNeXt-Tiny** ⚡ (Modern CNN, competitive performance)  
4️⃣ **DINOv2 (ViT-Small)** 🎯 (Best self-supervised model)  
5️⃣ **MobileNetV3** ⚡⚡ (Fastest model, best for real-time inference)  

These models are integrated into an **interactive UI using Gradio** and deployed on **Hugging Face Spaces**.

---

## 🛠 **How to Use the Model?**

### **1️⃣ Train the Model & Extract Features**
Run the following command to train and extract image embeddings:

```bash
python train.py --model resnet18
