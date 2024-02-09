import streamlit as st  
import tensorflow as tf  
from PIL import Image
import numpy as np
CLASS_LABELS = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                   'Tomato___Early_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy']

@st.cache_resource


def load_model():
    model  = tf.keras.models.load_model('./models/best_model_50_fullset_multiple.h5')
    return model

with st.spinner('model is being loaded'):
    model = load_model()

st.write("""
          # plant disease classify
         """)

file  = st.file_uploader("please upload the image",type=['jpg','png'])

def import_and_predict(image_data,model):
    size = (256,256)
    image = Image.open(file)

    image = image.resize((size))
    image = np.array(image)
    img_reshape = image / 255.0
    img_reshape = np.expand_dims(img_reshape,axis=0)

    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text('please upload a file')
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    predictions_label = CLASS_LABELS[np.argmax(predictions[0])]
    st.write("disease name: ",predictions_label)
