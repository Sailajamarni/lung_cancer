import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# Load the pre-trained model for image classification
model = load_model('models/keras_model.h5')

# Function to classify the uploaded image
def classify_image(img):
    # Preprocess the image
    img = img.resize((224, 224))  # Resize the image to match the input size of the model
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values to [0, 1]

    # Predict the class of the image
    prediction = model.predict(img_array)

    # Assuming the model has two classes: 'lung' and 'non-lung'
    if prediction[0][0] >= 0.5:
        return 'lung'
    else:
        return 'non-lung'

# Streamlit app
st.title('Lung Image Classifier')

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Classify the uploaded image
    classification = classify_image(img)
    if classification == 'lung':
        st.success("The uploaded image is a lung image.")
    else:
        st.error("The uploaded image is not a lung image. Please upload a lung image.")
