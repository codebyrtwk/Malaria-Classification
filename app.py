import streamlit  as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import glob 
import os

model = load_model('model_vgg19.h5')

st.title("Malaria Detection")

#upload image
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
button = st.button("Predict")

if image_file is not None and button:
    image = Image.open(image_file)
    image_resize = image.resize((224, 224))
    img_array =np.asarray(image_resize)
    img_array= img_array/225
    x = np.expand_dims(img_array,axis=0)
    # image_data = preprocess_input(x)
    prediction = model.predict_classes(x)
    st.info(prediction)
    a = np.argmax(prediction)
    st.info(a)

    if(a == 1 ):
    	st.info("Infected")
    else:
    	st.info('UnInfected')
	
    st.image(image_resize, caption="Bacteria Image", use_column_width=True)

if image_file is None and button:
	st.info("You haven't uploaded any Image yet !!!")


