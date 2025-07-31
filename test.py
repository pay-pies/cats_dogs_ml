import numpy as np
from tensorflow import keras
import streamlit as st
from PIL import Image

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://hc-cdn.hel1.your-objectstorage.com/s/v3/04c50d2ba191391c200fc554cc0e40cbc78ad578_screenshot_2025-07-31_133349.png");
        background-size: cover; 
        background-position: center; 
        background-repeat: no-repeat; 
        background-attachment: fixed; 
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: monospace !important;
        color: #e5e4ca !important; 
    }

    div[data-testid="stText"], div[data-testid*="stMarkdownContainer"] {
        font-family: monospace !important;
        color: #e5e4ca !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #555555 !important;
        border: 2px dashed #e5e4ca !important;
        color: #e5e4ca !important; 
    }

    [data-testid="stFileUploaderDropzone"] p {
        font-family: monospace !important;
        color: #e5e4ca !important;
    }

    [data-testid="stFileUploaderDropzone"] svg {
        fill: #e5e4ca !important; 
    }

    [data-testid="stFileUploader"] small {
        color: #e5e4ca !important; 
        font-family: monospace !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #555555 !important; 
        color: #e5e4ca !important;
        font-family: monospace !important;
        border: 1px solid #e5e4ca !important; 
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #777777 !important; 
    }
    
    img[data-testid="stLogo"] {
        height: 100px;
        width: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


model_path = "cats_dogs_sequential_model.keras" 
model = keras.models.load_model(model_path)

target_size = (128, 128) 

st.logo("https://hc-cdn.hel1.your-objectstorage.com/s/v3/6bb3114e4a06cf8107e9cf73791c3b2534957036_logo.png")
st.title('Cats vs Dogs Image Classification')
st.text('Note: This model has an accuracy of about 0.85 on the test set so it may not be perfect! Only JPG images are supported, sorry.)')
photo = st.file_uploader("Upload an image of a cat or dog", type=["jpg"])

if photo is not None: 
    img = Image.open(photo)  
    img = img.resize(target_size)  
    img_array = np.array(img, dtype=np.float32) / 255.0  

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    st.write(f"The image is {(100-(100*pred)):.2f}% likely to be a cat and {(100*pred):.2f}% likely to be a dog.")
    if pred > 0.75:
        st.write("Therefore, this image is a Dog!")
    elif pred > 0.5:
        st.write("The model is not very confident about this image, but it is more likely to be a dog.")
    elif pred > 0.25:
        st.write("The model is not very confident about this image, but it is more likely to be a cat.")
    else:
        st.write("Therefore, this image is a Cat!")
