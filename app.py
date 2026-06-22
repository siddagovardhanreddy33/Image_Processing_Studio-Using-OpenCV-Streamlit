import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from utils.filters import (grayscale,blur,threshold,negative,sepia)
from utils.enhancement import (brightness_contrast,sharpen)
from utils.edge_detection import edge_detect
from utils.augmentation import ( horizontal_flip,vertical_flip, rotate, add_noise)



st.set_page_config(layout="wide")
st.title("📷 Image Preprocessor using OpenCV")

# Upload image
uploaded_file = st.file_uploader("Upload Image",["jpg", "jpeg", "png"])

if uploaded_file:
    img = np.array(Image.open(uploaded_file))
    processed = img.copy()

    # Resize
    st.sidebar.header("Resize")
    width = st.sidebar.slider("Width",100,1000,img.shape[1])
    height = st.sidebar.slider("Height",100,1000,img.shape[0])
    processed = cv2.resize(processed,(width, height)    )





    # Enhancement
    st.sidebar.header("Enhancement")
    brightness = st.sidebar.slider("Brightness",-100,100,0)
    contrast = st.sidebar.slider( "Contrast",0.5,3.0,1.0)
    processed = brightness_contrast(processed,brightness,contrast)

    # Sharpen checkbox
    sharpen_image = st.sidebar.checkbox("Sharpen Image")
    if sharpen_image:
        processed = sharpen(processed)







    # Filters
    st.sidebar.header("Filters")
    filter_option = st.sidebar.selectbox("Choose Filter",
        [
            "Original",
            "Grayscale",
            "Blur",
            "Threshold",
            "Negative",
            "Sepia"
        ]
    )

    if filter_option == "Grayscale":
        processed = grayscale(processed)
    elif filter_option == "Blur":
        processed = blur(processed)
    elif filter_option == "Threshold":
        processed = threshold(processed)
    elif filter_option == "Negative":
        processed = negative(processed)
    elif filter_option == "Sepia":
        processed = sepia(processed)




    # Edge Detection

    edge = st.sidebar.checkbox("Edge Detection")
    if edge:
        processed = edge_detect(processed)

    

    # Augmentation

    st.sidebar.header("Augmentation")
    augmentation_option = st.sidebar.selectbox("Choose Augmentation",
        [
            "None",
            "Horizontal Flip",
            "Vertical Flip",
            "Rotate",
            "Add Noise"
        ]
    )

    if augmentation_option == "Horizontal Flip":
        processed = horizontal_flip(processed)
    elif augmentation_option == "Vertical Flip":
        processed = vertical_flip(processed)
    elif augmentation_option == "Rotate":
        processed = rotate(processed)
    elif augmentation_option == "Add Noise":
        processed = add_noise(processed)





    # Normalize
    normalize = st.sidebar.checkbox("Normalize (0-1)")
    if normalize:
        processed = processed.astype(np.float32) / 255.0






    # Display Images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img,use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(processed,use_container_width=True)






    # Download Image

    if processed.dtype == np.float32:
        save_img = (processed * 255).astype(np.uint8)
    else:
        save_img = processed.astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(save_img).save(buffer,format="PNG")

    st.download_button("⬇ Download Image",buffer.getvalue(),"processed_image.png","image/png")