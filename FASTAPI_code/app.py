from fastapi import FastAPI, Request, File
from pydantic import BaseModel
from typing import List
import numpy as np
from io import BytesIO
import cv2
import base64
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from skimage import feature
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

class ImageData(BaseModel):
    image: bytes


# model_loaded = load_model('D:/batul/Sem2/dip/Project/Lead_Disease/BTL_model_leaf')

#Define a function to attempt loading the model multiple times
def attempt_load_model(filepath, max_retries=3):
    for attempt in range(max_retries):
        try:
            model = load_model(filepath)
            return model
        except OSError as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise e
try:
    model = attempt_load_model('../Batul_leaf.h5')
except OSError as e:
    print("Failed to load the model after multiple attempts.")

classes = ['Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy', 'Background_without_leaves', 'Blueberry_healthy',
           'Cherry_powdery_mildew', 'Cherry_healthy', 'Corn_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy',
           'Grape_black_rot', 'Grape_black_measles', 'Grape_leaf_blight', 'Grape_healthy', 'Orange_haunglongbing', 'Peach_bacterial_spot',
           'Peach_healthy', 'Pepper_bacterial_spot', 'Pepper_healthy', 'Potato_early_blight', 'Potato_healthy', 'Potato_late_blight',
           'Raspberry_healthy', 'Soybean_healthy', 'Squash_powdery_mildew', 'Strawberry_healthy', 'Strawberry_leaf_scorch', 'Tomato_bacterial_spot',
           'Tomato_early_blight', 'Tomato_healthy', 'Tomato_late_blight', 'Tomato_leaf_mold', 'Tomato_septoria_leaf_spot',
           'Tomato_spider_mites_two-spotted_spider_mite', 'Tomato_target_spot', 'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus']

@app.post("/predict")
async def predict(file: UploadFile):
    # # Extract the image data from the request
    image_bytes = await file.read()
    image1 = Image.open(io.BytesIO(image_bytes))

    # Load the image with the target size (replace with the size your model expects)
    img = image.load_img(io.BytesIO(image_bytes), target_size=(32, 32))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the shape the model expects (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Optionally, normalize the image (if your model expects normalized input)
    # For example, if the model was trained with images normalized to the range [0, 1]:
    img_array = img_array / 255.0
    # Perform prediction using your model
    predictions = model.predict(img_array).tolist()
    # Return the prediction result
    return {"prediction": classes[predictions[0].index(max(predictions[0]))]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)