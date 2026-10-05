
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NLP Text Classifier",
    page_icon="📝",
    layout="centered"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = tf.keras.models.load_model(
    "nlp_text_classifier_glove_bigru.keras"
)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# --------------------------------------------------
# Load label encoder
# --------------------------------------------------

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# --------------------------------------------------
# Load maximum sequence length
# --------------------------------------------------

with open("max_len.pkl", "rb") as file:
    MAX_LEN = pickle.load(file)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("📝 NLP Text Classifier")

st.write(
    "Enter a text below and the model will predict its class."
)


# --------------------------------------------------
# Text input
# --------------------------------------------------

text = st.text_area(
    "Enter your text:",
    placeholder="Type your text here..."
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Convert text into token IDs
        sequence = tokenizer.texts_to_sequences([text])

        # Apply the same padding used during training
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        # Make prediction
        probabilities = model.predict(
            padded_sequence,
            verbose=0
        )[0]

        # Get predicted class
        predicted_index = np.argmax(probabilities)

        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # Get confidence
        confidence = probabilities[predicted_index] * 100


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.success(
            f"Predicted Class: {predicted_label}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )
