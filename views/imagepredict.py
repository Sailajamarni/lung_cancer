import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model1= tf.keras.models.load_model('lung_image_prediction_model.keras')
    return model1

model1 = load_model()

# Make predictions
def predict_image(image, model):
    # Resize the image to match the input size of the model
    img = image.resize((224, 224)).convert('RGB')
    # Convert image to RGB mode if it's not already in that mode
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title('Lung Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image, model)

    if prediction < 0.5:
        st.write("This image contains a lung.")
    else:
        st.write("This image does not contain a lung.")
        
