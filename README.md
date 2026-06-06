# 🔬 Skin Cancer Detection using Deep Learning

A deep learning-based multi-class skin lesion classifier built with **EfficientNetB3** and transfer learning, trained on the **HAM10000 dataset**. The model classifies skin lesion images into 7 categories to assist in early skin cancer detection.

---

## 📌 Project Overview

Skin cancer is one of the most common cancers worldwide, and early detection significantly improves survival rates. This project uses a pre-trained convolutional neural network (EfficientNetB3) with custom classification layers to identify 7 types of skin lesions from dermoscopy images.

---

## 🗂️ Dataset — HAM10000

- **Source**: [Kaggle - Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Size**: 10,015 labeled dermoscopy images
- **Classes**: 7 skin lesion types

| Code | Condition |
|------|-----------|
| `nv` | Melanocytic Nevi |
| `mel` | Melanoma |
| `bkl` | Benign Keratosis-like Lesions |
| `bcc` | Basal Cell Carcinoma |
| `akiec` | Actinic Keratoses |
| `vasc` | Vascular Lesions |
| `df` | Dermatofibroma |

---

## 🧠 Model Architecture

Transfer learning on **EfficientNetB3** with `noisy-student` pretrained weights.

```
EfficientNetB3 (pretrained, frozen base)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense (64, ReLU)
        ↓
Dense (7, Softmax)  ← Output: 7 class probabilities
```

---

## ⚙️ Tech Stack

## ⚙️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Model building and training
- **EfficientNetB3** – Pretrained CNN backbone
- **NumPy** – Numerical operations
- **Pandas** – Data loading and analysis
- **Matplotlib & Seaborn** – Data visualization
- **PIL (Pillow)** – Image loading and resizing
- **Scikit-learn** – Data splitting and evaluation metrics
- **Google Colab** – Training environment
---

## 🔄 Workflow

### 1. Data Analysis
- Explored class distribution and identified class imbalance
- Handled missing values in the `age` column (filled with mean)
- Visualized sample images from each class

### 2. Image Preprocessing
- Resized all images from 450×600 to **90×120 pixels**
- Applied **standardization** (zero mean, unit variance) on pixel values

### 3. Data Splitting
- 99% Training / 1% Test
- Further split training into 85% Train / 15% Validation

### 4. Training
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, Recall
- **Epochs**: 20 | **Batch Size**: 16
- **Data Augmentation**: Random rotation (±10°), random zoom (10%)
- **Callback**: ReduceLROnPlateau — halves learning rate if val_loss stagnates for 3 epochs

### 5. Evaluation
- Classification report (Precision, Recall, F1-score per class)
- Confusion matrix
- Training vs Validation accuracy and loss plots

---

## 📊 Results

- Evaluated on held-out test set using accuracy, recall, and confusion matrix
- Model saved as `skin_cancer_efficientnetb3.h5`

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Install dependencies
```bash
pip install tensorflow keras efficientnet numpy pandas matplotlib seaborn pillow scikit-learn
```

### 3. Set up Kaggle API
- Upload your `kaggle.json` API key
- The notebook will automatically download the HAM10000 dataset

### 4. Run the notebook
Open `SKIN_CANCER_PREDICTION_Final.ipynb` in **Google Colab** and run all cells.

---

## 📁 Project Structure

```
skin-cancer-detection/
│
├── SKIN_CANCER_PREDICTION_Final.ipynb   # Main notebook
├── skin_cancer_efficientnetb3.h5        # Saved model (after training)
└── README.md
```

---

## 🔮 Sample Prediction

After training, the model takes a new skin lesion image, preprocesses it, and predicts the lesion type:

```python
# Resize and normalize the input image
test_image = np.asarray(image.resize((120, 90))).reshape(1, 90, 120, 3)

# Predict
prediction = model.predict(test_image)
predicted_class = lesion_classes_dict[np.argmax(prediction)]
print(predicted_class)  # e.g., "Melanoma"
```


## ⚠️ Disclaimer

This project is built for educational purposes only. It is not intended for clinical or diagnostic use.