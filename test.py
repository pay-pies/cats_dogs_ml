#import os
import numpy as np
from tensorflow import keras
import streamlit as st
#from tensorflow.keras.preprocessing import image
from PIL import Image
#from tensorflow.keras.utils import load_img, img_to_array

model_path = "cats_dogs_sequential_model.keras" 
model = keras.models.load_model(model_path)

target_size = (128, 128) 

st.title('Cats vs Dogs Image Classification')
st.text('Note: This model has an accuracy of about 0.85 on the test set so it may not be perfect! Only JPG images are supported, sorry.)')
photo = st.file_uploader("Upload an image of a cat or dog", type=["jpg"])

if photo is not None: #hellooo
    img = Image.open(photo)  
    img = img.resize(target_size)  
    img_array = np.array(img, dtype=np.float32) / 255.0  

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    st.write("The image is " + str(100-(100*pred)) + '%' + " likely to be a cat and "+ str(100*pred) + '%'" likely to be a dog.")
    if pred > 0.5:
        st.write("Therefore, this image is a Dog!")
    else:
        st.write("Therefore, this image is a Cat!")

# img_folder = 'C:\\Users\\Halley\\Documents\\Projects\\Summer of Making\\cats_dogs_ml\\PetImages\\Test'


# for test_image in os.listdir(img_folder): 
#     img_path = os.path.join(img_folder, test_image)
# # Load the image and resize it
#     img = load_img(img_path, target_size=target_size)

#     img_array = np.array(img)

#     # Expand dimensions to create a batch of 1 (batch_size, height, width, channels)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Normalize pixel values (e.g., to [0, 1] if that's what you did during training)
#     # If your training data was scaled to [0, 1] by dividing by 255.0
#     img_array = img_array / 255.0

#     predictions = model.predict(img_array)

#     if predictions[0] > 0.5:
#         st.write(test_image + " is a Dog!")
#     else:
#         st.write(test_image + " is a Cat!")