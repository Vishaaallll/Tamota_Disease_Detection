# 🍅 Tomato Disease Detection

A deep learning project that classifies tomato leaf images as healthy or diseased using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## 📌 Overview

Plant diseases significantly impact crop yield and quality. This project uses computer vision to automatically detect diseases in tomato plants from leaf images, enabling early diagnosis and helping farmers take timely action.

The model is trained on labeled images of tomato leaves and predicts the disease category (or healthy status) for a given input image.

## 🎯 Features

- CNN-based image classification for tomato leaf disease detection
- Multi-class classification across several disease categories
- Trained and evaluated using TensorFlow/Keras
- Achieves reliable accuracy on validation/test data
- Simple interface for testing predictions on new images

## 🧠 Model Architecture

- **Framework:** TensorFlow / Keras
- **Model type:** Convolutional Neural Network (CNN)
- **Input:** RGB tomato leaf images (resized to a fixed input shape)
- **Layers:** Convolution + Pooling blocks → Flatten → Dense layers → Softmax output
- **Loss function:** Categorical Cross entropy
- **Optimizer:** Adam

> Update this section with your exact architecture (number of conv layers, filter sizes, dropout, etc.) once finalized.

## 🗂️ Dataset

- **Source:** *( Tomato leaf disease detection - https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)*
- **Classes:** *(Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Milte TWO-Spotted, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus)*
- **Split:** Train / Validation / Test
- **Preprocessing:** Image resizing, normalization, and augmentation (rotation, flip, zoom)

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | *96.8888* |
| Validation Accuracy | *94.325* |
| Loss | *0.05* |

> Replace the placeholders above with your actual evaluation metrics.

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook / Google Colab (for training)

## 📁 Project Structure

```
tomato-disease-detection/
├── dataset/                # Training and validation images
├── notebooks/               # Jupyter notebooks for EDA and training
├── model/                   # Saved trained model (.h5 / .keras)
├── src/
│   ├── train.py             # Model training script
│   ├── predict.py           # Inference script
│   └── preprocess.py        # Data preprocessing utilities
├── requirements.txt
└── README.md
```

> Adjust this structure to match your actual repo layout.

## 🚀 Getting Started

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training the Model

```bash
python src/train.py
```

### Running Predictions

```bash
python src/predict.py --image path/to/leaf_image.jpg
```

## 📈 Future Improvements

- Expand dataset with more disease classes and real-world images
- Deploy as a web app (Flask/Streamlit) or mobile app for field use
- Use transfer learning (e.g., MobileNet, EfficientNet) to improve accuracy
- Add Grad-CAM visualization for model explainability

## 👤 Author

**Vishal Yadav **
B.Tech Computer Science

## 📄 License

This project is licensed under the MIT License.
