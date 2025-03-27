import cv2
import numpy as np
import streamlit as st 
from io import BytesIO
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from ultralytics import YOLO
import matplotlib.pyplot as plt

from PIL import Image

inference_transform = A.Compose( 
	[ 
		A.LongestMaxSize(max_size=1024),
		A.PadIfNeeded( 
			min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT
		),
		A.Normalize(
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		), 
		ToTensorV2() 
	],
) 
models = {
    'NOVEL': 'models/novel model.pt', "YOLOv5n":'models/best_v5n.pt',
    # "YOLOv5m":'models/best_v5m.pt', "YOLOv5x":'models/best_v5x.pt',
    # "YOLOv8n":'models/best_v8n.pt', "YOLOv8m":'models/best_v8m.pt',
    # "YOLOv8x":'models/best_v8x.pt',
	"YOLO11n":'models/best_11n.pt',
    # "YOLO11m":'models/best_11m.pt', "YOLO11x":'models/best_11x.pt'
    }

def main():
    # st.title("Potatoes Leaves Classifier")
    html_temp = """
    <div style="background-color:purple;padding:10px">
    <h2 style="color:white;text-align:center;"> Leaves Diseases Detection App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    byteImage = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

    if byteImage:
        bytesImg = BytesIO(byteImage.getvalue())
        image = Image.open(bytesImg)
        img_array= np.array(image)
        augs = inference_transform(image =img_array)

        #display image
        st.image(image)

    # names = 
    model_name = st.selectbox('Select model to be used:',([a for a,_ in models.items()]))
    model = YOLO(models[model_name], task = 'detect')

    if st.button("Detect") and image:
        results = model(augs['image'].unsqueeze(dim = 0))
        res = results[0].plot()
        st.image(Image.fromarray(res))
        

if __name__=='__main__':
    main()
