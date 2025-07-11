# 🛰️ Satellite Image Classifier (DINOv2 + SVM)

A high-performance remote sensing image classifier leveraging the power of self-supervised visual transformers (**DINOv2**) combined with a lightweight **Support Vector Machine (SVM)** for downstream classification.

> 🔗 **Live Demo**: [Streamlit App](https://dinov2-svm-satellite-classifier-jeyguhu5uhlnhmgtkqlrpe.streamlit.app/)

---

## Overview

This project classifies satellite and GIS imagery into defined categories using learned embeddings from Meta AI's **DINOv2 ViT-S/14 model**, followed by an **SVM classifier** trained on top of those embeddings.

The goal is to enable **real-time, accurate classification** of remote sensing data with a lightweight, interpretable model pipeline.

---

## 🧠 Model Architecture

- **Feature Extractor**: `DINOv2 ViT-S/14` from a locally cloned FacebookResearch repo.
- **Classifier**: `scikit-learn SVM` trained on DINOv2 embeddings.
- **Deployment**: Streamlit UI for interactive uploads and real-time predictions.

---

## 🛰️ Use Case

- Classifying **satellite** or **aerial imagery** for:
  - Land use & land cover (LULC)
  - Urbanization tracking
  - Environmental monitoring
  - Agriculture & forestry segmentation

---

## 🚀 How It Works

1. **User uploads an image**
2. **DINOv2** generates high-dimensional visual embeddings
3. **SVM classifier** predicts the class from the learned embedding
4. **Results are displayed instantly** with class label and visualization

---

## 🧪 Confusion Matrix (Validation)

Below is the confusion matrix on the held-out test set, indicating per-class prediction performance.

![Confusion Matrix](confusion_matrix.png)

---

## 🛠️ Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/inayatrahimdev/DINOv2-SVM-Satellite-Classifier.git
cd DINOv2-SVM-Satellite-Classifier
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```
---

## 📂 Project Structure

```
├── app.py                     # Streamlit frontend + classifier pipeline
├── dino_svm_classifier.pkl    # Trained SVM model
├── class_names.npy            # Class label mapping
├── dinov2_repo/               # Local DINOv2 repo (hubconf.py + model checkpoints)
├── confusion_matrix.png       # Model performance visualization
├── requirements.txt           # Python dependencies
└── README.md                  # You're reading it
```

---

## 📌 Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* Streamlit
* torchvision
* scikit-learn
* numpy
* Pillow
* joblib

---

## ✍️ Citation

If you use this codebase in research or teaching, consider citing the original DINOv2 paper:

```
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Hugo and Moutakanni, Theo and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

---

## 📬 Contact

Built by [Inayat Rahim](https://github.com/inayatrahimdev)
For collaborations or academic use, feel free to open an issue or reach out via email.

---

## 🧭 License

This project is released under the MIT License.

```
