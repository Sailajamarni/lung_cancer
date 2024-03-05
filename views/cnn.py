import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
model = load_model('models/keras_model.h5')

model1=load_model('models/lung_image_prediction_model.keras')


def load_view():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)  

    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    @st.cache_data()
    def loading_model():
        fp = "models/keras_model.h5"
        model_loader = load_model(fp)
        return model_loader
    # def load_lmodel():
    #     f1="models/lung_image_prediction_model.keras"
    #     model_load=load_model(f1)
    #     return model_load
    def classify_image(img):
        # Preprocess the image
        img = img.resize((224, 224))  # Resize the image to match the input size of the model
        img = img.convert('RGB') 
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.  # Normalize pixel values to [0, 1]

        # Predict the class of the image
        prediction = model1.predict(img_array)
        if prediction < 0.5:
            return 'lung'
        else:
            return 'non-lung'
        

    cnn = loading_model()
    st.write("""# Lung Cancer Detection using CNN and CT-Scan Images""")

    temp = st.file_uploader("Upload CT-Scan Image", type=['png', 'jpeg', 'jpg'])
    # classification = classify_image()
    # if classification != 'lung':
    #         st.error("The uploaded image is a lung image.")
    if temp is not None:
        file_details = {"FileName": temp.name, "FileType": temp.type, "FileSize": temp.size}
        st.write(file_details)

    buffer = temp
    temp_file = NamedTemporaryFile(delete=False)
    
    if buffer:
        temp_file.write(buffer.getvalue())
        st.write(image.load_img(temp_file.name))
    






    if buffer is None:
        st.text("Please upload an image file")
        

    else:
            ved_img =image.load_img(temp_file.name, target_size=(224, 224))
            pp_ved_img = img_to_array(ved_img)
            pp_ved_img = pp_ved_img / 255
            pp_ved_img = np.expand_dims(pp_ved_img, axis=0)
            classification = classify_image(ved_img )
            if classification != 'lung':
                st.error("The uploaded image is not a lung image.")
            # predict
            else:
                hardik_preds = cnn.predict(pp_ved_img)
                print(hardik_preds[0])
                if hardik_preds[0][0] >= 0.5:
                    out = ('I am {:.2%} percent confirmed that this is a Normal Case'.format(hardik_preds[0][0]))
                    st.balloons()
                    st.success(out)
                else: 
                    out = ('I am {:.2%} percent confirmed that this is a Lung Cancer Case'.format(1 - hardik_preds[0][0]))
                    st.error(out)

                img_display = Image.open(temp_file.name)
                st.image(img_display, use_column_width=True)

        # Call the load_view function
