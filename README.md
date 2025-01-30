# 🔍 Image Similarity Search using Deep Learning

This project implements a **content-based image retrieval (CBIR) system** using deep learning models to find visually similar images. The system extracts feature embeddings from pre-trained models and performs **FAISS-based similarity search**.

---
## 🔍 Similarity Search Results

Here are **5 example similarity search results** from our FAISS-based image retrieval system:

### 🖼 Query 1
![Query 1](https://github.com/manojrathor08/image_similarity/blob/main/similarity_results/query_0.png)

---

### 🖼 Query 2
![Query 2](https://github.com/manojrathor08/image_similarity/blob/main/similarity_results/query_2.png)

---

### 🖼 Query 3
![Query 3](https://github.com/manojrathor08/image_similarity/blob/main/similarity_results/query_3.png)

---

### 🖼 Query 4
![Query 4](https://github.com/manojrathor08/image_similarity/blob/main/similarity_results/query_4.png)

---

### 🖼 Query 5
![Query 5](https://github.com/manojrathor08/image_similarity/blob/main/similarity_results/query_5.png)

---


## 📂 Dataset Information

We use the **Fashion Product Images Small** dataset from Hugging Face 🤗:

- **Dataset Source:** [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- **Total Images:** **44,072**
- **Total Classes:** **45**
- **Selected Classes:** **Top 10 most frequent classes**
- **Data Split:** Stratified split into **Training, Validation, and Test sets**.

## **📊 Data Splitting Strategy**
The dataset was divided in a **stratified manner** to maintain class balance:
- **Train Set:** **70%** of the selected data
- **Validation Set:** **15%** (used for evaluation)
- **Test Set:** **15%** (used for final model testing)

---

## 🚀 Project Workflow

### 1️⃣ Data Preprocessing (`data_preparation.py`)
- Load images and labels from **Hugging Face Dataset**.
- **Filter top 10 classes** from the original **45 classes**.
- **Split data into Train, Validation, and Test sets**.
- Save images and **encode labels** for training.

### 2️⃣ Feature Extraction & Indexing (`train.py`)
- Extract embeddings using **pre-trained deep learning models**.
- **Models Evaluated:** **ResNet-18, EfficientNet-B0, MobileNetV3, SqueezeNet, ShuffleNet-V2, ConvNeXt-Tiny**.
- Store image embeddings in a **FAISS index** for fast retrieval.

### 3️⃣ Similarity Search & Evaluation (`test.py`)
- Load the **FAISS index** and validation data.
- Search for **top 5 most similar images**.
- Evaluate retrieval using **Precision@5, Retrieval Accuracy, Weighted Precision, and Mean Reciprocal Rank (MRR)**.

### 4️⃣ UI Deployment on Hugging Face Spaces
- Built an **interactive Gradio UI**.
- Users can **upload an image** to retrieve **5 most similar images** from the dataset.

---

## 🛠 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/fashion-image-retrieval.git
cd fashion-image-retrieval 
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt 
```

### 3️⃣ Prepare the Data
```bash
python data_preparation.py 
```

### 4️⃣ Train the Model & Build FAISS Index
```bash
python train.py 
```

### 5️⃣ Evaluate the Model
```bash
python test.py 
```

### 6️⃣ Run the Interactive Gradio UI
```bash
python app.py 
```


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

---
## 🤝 Contributing
Feel free to **open an issue or submit a PR** if you’d like to contribute. 🚀

---

## 💡 Acknowledgments
This project is built upon numerous **open-source models and libraries**. We acknowledge the following research papers and frameworks that contributed to the development of this work:

### 📚 **Dataset**
- **Fashion Product Images Small**: Hugging Face dataset [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small)

### 🔍 **FAISS - Fast Similarity Search**
- Johnson, J., Douze, M., & Jégou, H. (2019).  
  **"Billion-scale similarity search with GPUs"**.  
  *IEEE Transactions on Big Data*.  
  📄 [FAISS Paper](https://arxiv.org/abs/1702.08734)

### 🏗 **Pretrained Models Used**
We utilized several **deep learning models** for feature extraction. Below are the original research papers for each model:

- **ResNet-18**:  
  He, K., Zhang, X., Ren, S., & Sun, J. (2016).  
  **"Deep Residual Learning for Image Recognition"**.  
  *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.  
  📄 [Paper](https://arxiv.org/abs/1512.03385)

- **EfficientNet-B0**:  
  Tan, M., & Le, Q. V. (2019).  
  **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"**.  
  *Proceedings of the International Conference on Machine Learning (ICML)*.  
  📄 [Paper](https://arxiv.org/abs/1905.11946)

- **MobileNetV3**:  
  Howard, A., Sandler, M., Chu, G., et al. (2019).  
  **"Searching for MobileNetV3"**.  
  *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.  
  📄 [Paper](https://arxiv.org/abs/1905.02244)

- **SqueezeNet**:  
  Iandola, F., Han, S., Moskewicz, M., et al. (2016).  
  **"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"**.  
  *arXiv preprint*.  
  📄 [Paper](https://arxiv.org/abs/1602.07360)

- **ShuffleNet-V2**:  
  Ma, N., Zhang, X., Zheng, H., & Sun, J. (2018).  
  **"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"**.  
  *Proceedings of the European Conference on Computer Vision (ECCV)*.  
  📄 [Paper](https://arxiv.org/abs/1807.11164)

- **ConvNeXt-Tiny**:  
  Liu, Z., Mao, H., Wu, C., et al. (2022).  
  **"A ConvNet for the 2020s"**.  
  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.  
  📄 [Paper](https://arxiv.org/abs/2201.03545)

### 💡 **Deep Learning Frameworks**
This project relies on the following **deep learning and machine learning libraries**:
- **PyTorch**: [Website](https://pytorch.org/)  
- **Torchvision**: [GitHub](https://github.com/pytorch/vision)  
- **FAISS**: [GitHub](https://github.com/facebookresearch/faiss)  
- **Scikit-learn**: [Website](https://scikit-learn.org/)  
- **Hugging Face Datasets**: [Website](https://huggingface.co/datasets)  










