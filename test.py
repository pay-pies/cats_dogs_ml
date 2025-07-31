import numpy as np
from tensorflow import keras
import streamlit as st
from PIL import Image

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
