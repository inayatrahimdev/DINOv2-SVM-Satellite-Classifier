# ✅ FIRST Streamlit command — must be at the very top
import streamlit as st
st.set_page_config(page_title="Satellite Image Classifier", layout="wide")

# ✅ Other imports
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import joblib

# ✅ Load SVM Classifier and Class Names
clf = joblib.load("dino_svm_classifier.pkl")
class_names = np.load("class_names.npy", allow_pickle=True)

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load DINOv2 model from local repo (fixed to use your local dinov2_repo)
@st.cache_resource
def load_dino():
    model = torch.hub.load(
        repo_or_dir="dinov2_repo",        # ✔️ Path to your local cloned repo
        model="dinov2_vits14",            # ✔️ The model name from hubconf.py
        source="local"                    # ✔️ Local source to avoid GitHub pull
    )
    model.eval()
    model.to(device)
    return model

# ✅ Load model
dino = load_dino()

# ✅ Image Transform
transform = T.Compose([
    T.Resize(244),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

# ✅ Streamlit UI
st.title("🌍 Satellite / GIS Image Classifier (DINOv2 + SVM)")
st.markdown("Upload any remote sensing or satellite image to classify it using a trained DINOv2 + SVM model.")

# ✅ File Upload
uploaded = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])

# ✅ Embedding + Prediction
def get_embedding(img_pil):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = dino(img_tensor)
    return features.cpu().numpy().reshape(1, -1)

# ✅ Handle Uploaded Image
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    
    with st.spinner("🔍 Classifying..."):
        embedding = get_embedding(image)
        pred = clf.predict(embedding)[0]
        st.success(f"✅ Predicted Class: **{class_names[pred]}**")
