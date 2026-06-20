import os
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# Streamlit Deployment App
# Safety Helmet Compliance Detection
# Classes: person, helmet, no_helmet
# ============================================================

st.set_page_config(
    page_title="Safety Helmet Compliance Detection",
    page_icon="🦺",
    layout="centered"
)

st.title("Safety Helmet Compliance Detection")
st.write(
    "Upload an image to detect construction workers, safety helmets, "
    "and non-compliance conditions using YOLO."
)

st.markdown(
    """
    **Detected classes:**
    - person
    - helmet
    - no_helmet
    """
)

# Model path
MODEL_PATH = "best.pt"


@st.cache_resource
def load_model(model_path):
    """
    Load YOLO model once and cache it for faster inference.
    """
    model = YOLO(model_path)
    return model


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file `best.pt` was not found. "
        "Please place your trained YOLO model file in the same folder as app.py."
    )
    st.stop()


model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a construction-site image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_image_path = temp_file.name

    st.subheader("Detection Result")

    with st.spinner("Running detection..."):
        results = model.predict(
            source=temp_image_path,
            conf=confidence,
            imgsz=640
        )

    result_image = results[0].plot()
    st.image(result_image, use_container_width=True)

    st.subheader("Detected Objects")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        class_names = model.names

        detection_data = []

        for box in boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])
            class_name = class_names[class_id]

            detection_data.append(
                {
                    "Class": class_name,
                    "Confidence": round(confidence_score, 3)
                }
            )

        st.table(detection_data)
    else:
        st.warning("No object detected in the uploaded image.")

    os.remove(temp_image_path)
else:
    st.info("Please upload an image to start detection.")