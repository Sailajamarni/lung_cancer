# import base64
import pickle
import streamlit as st
import base64
import utils as utl
from views import introduction,about,prediction,cnn
 

st.set_page_config(page_title='Lung Cancer Detection')
#Loading models
cancer_model = pickle.load(open('models/final_model.sav', 'rb'))


st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()


def navigation():
    route = utl.get_current_route()
    if route == "introduction":
        introduction.load_view()
    elif route == "about_dataset":
        about.load_view()
    elif route == "prediction":
        prediction.load_view()
    elif route == "cnn_based":
        cnn.load_view()
    elif route == None:
        introduction.load_view()
        
navigation()






st.markdown(
    f"""
    <style>
        .stApp{{
            background-color: yellow;
        }}
    </style>
    """,
    unsafe_allow_html=True
)






# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# img = get_img_as_base64("images/liver.png")

# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://cdn.sanity.io/images/0vv8moc6/chroma/d7fc8efcea131de9932ecfcc91f712f7b39fb3e1-3466x3204.jpg");
# background-size: 100%;
# background-position: top-right;
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img}");
# background-size: 120%;
# background-position: center; 
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"] {{
# right: 2rem;
# }}
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True)

