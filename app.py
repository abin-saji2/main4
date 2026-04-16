import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import tensorflow as tf
import os

# ---------------- LOAD MODEL ----------------
model_path = "kidney_model.h5"

if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()

model = tf.keras.models.load_model(model_path, compile=False)

classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Kidney AI", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Kidney AI")
page = st.sidebar.radio("Navigation", ["🏠 Prediction", "ℹ️ About"])

# ---------------- PDF FUNCTION ----------------
def create_pdf(name, age, gender, state, phone, result, confidence):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Kidney Disease Report", ln=True, align='C')

    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, f"State: {state}", ln=True)
    pdf.cell(200, 10, f"Phone: {phone}", ln=True)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Prediction Result:", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Condition: {result}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, "Note: This is an AI-based prediction and not a medical diagnosis.")

    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path

# ================== HOME PAGE ==================
if page == "🏠 Prediction":

    st.markdown("""
    <h1 style='text-align:center;'>🩺 Kidney Disease Detection</h1>
    <p style='text-align:center;'>Upload CT Scan • AI Prediction • Download Report</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col2:
        state = st.text_input("State")
        phone = st.text_input("Phone")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze"):
            if not name or not phone:
                st.warning("⚠️ Please fill all details")
                st.stop()

            # Preprocess image
            img_resized = img.resize((128,128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            pred = model.predict(img_array)
            confidence = float(np.max(pred) * 100)
            result = classes[np.argmax(pred)]

            with col2:
                st.subheader("Prediction Result")

                # Chart
                df = pd.DataFrame({
                    "Condition": classes,
                    "Probability": pred[0]*100
                })
                st.bar_chart(df.set_index("Condition"))

                if confidence < 70:
                    st.warning("⚠️ Low confidence prediction")
                else:
                    st.success(f"🧾 Result: {result}")
                    st.info(f"Confidence: {confidence:.2f}%")

                    info = {
                        "Cyst": "Fluid-filled sac in kidney.",
                        "Normal": "No abnormality detected.",
                        "Stone": "Hard mineral deposits.",
                        "Tumor": "Abnormal growth."
                    }

                    st.write(info[result])

                    # PDF download
                    pdf_file = create_pdf(name, age, gender, state, phone, result, confidence)

                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Download PDF", f, "report.pdf")

    st.markdown("---")
    st.warning("⚠️ This is not a medical diagnosis")

# ================== ABOUT PAGE ==================
elif page == "ℹ️ About":

    st.title("About")

    st.write("""
    AI-based kidney disease detection system.

    - Uses Deep Learning (CNN)
    - Predicts from CT scan images
    - Generates PDF reports
    """)