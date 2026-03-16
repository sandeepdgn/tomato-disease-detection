# 🍅 Tomato Leaf Disease Detection & Classification

A deep learning pipeline for identifying tomato leaf diseases from images. The project focuses on high accuracy with efficient training using a custom CNN and a streamlined TensorFlow `tf.data` pipeline.

## ✨ Highlights
- **Multi-class classification** across 10 tomato leaf conditions.
- **Efficient training** with on-the-fly augmentation and dataset prefetching.
- **Resource-aware design** for running in constrained environments.

## 📌 Dataset
The notebook expects an image dataset in `dataset/` (not tracked in git). The directory should contain one folder per class:

```
dataset/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites/
├── Tomato_YellowLeaf__Curl_Virus/
├── Tomato__Target_Spot/
├── Tomato_healthy/
└── Tomato_mosaic_virus/
```

From the notebook output, the dataset contains **10 classes** and **16,011 images**.

## 🧠 Model & Training
The training pipeline is implemented in **`notebook/training.ipynb`** and includes:

- **Image size:** 256×256
- **Batch size:** 32
- **Epochs:** 50
- **Split:** 80% train / 10% validation / 10% test
- **Architecture:** Sequential CNN with multiple `Conv2D + MaxPooling` blocks followed by dense layers
- **Augmentation:** Random flips and rotations applied on-the-fly

## 📈 Results
The model achieves approximately **98% validation accuracy** on the dataset (as reported in the notebook).

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook / Google Colab
- TensorFlow (Keras), NumPy, Pandas, Matplotlib, OpenCV

### Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy pandas matplotlib opencv-python
```

### Run the Notebook
Open and execute the notebook in Jupyter or Colab:

```bash
jupyter notebook notebook/training.ipynb
```

## 🔍 Inference (After Training)
Once you have trained and exported the model to `models/`, you can load it in Python:

```python
import tensorflow as tf

model = tf.keras.models.load_model("models/your_saved_model")
```

The notebook includes a `predict()` helper that maps predictions to class labels.

## 🗂️ Project Structure
```
tomato-disease-detection/
├── notebook/
│   └── training.ipynb   # Training pipeline, EDA, and evaluation
├── models/              # Exported model files (.h5 / .keras) - ignored in git
├── dataset/             # Training data - ignored in git
├── .gitignore
└── README.md
```

## 🛣️ Roadmap
- FastAPI endpoint for real-time inference
- TensorFlow Lite conversion for mobile deployment
- Edge/IoT camera integration for greenhouse monitoring

## 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author
**Sandeep**  
GitHub: [@sandeepdgn](https://github.com/sandeepdgn)
