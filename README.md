# Leaf Disease Detection using Digital Image Processing and Deep Learning

## Index

1. [Abstract](#abstract)
   - [Problem Statement](#problem-statement)
   - [Objective](#objective)
   - [Data Description](#data-description)
2. [DIP Components and Deep Learning Tools](#dip-components-and-deep-learning-tools)
3. [Implementation](#implementation)
   - [Training Code](#training-code)
   - [Execution](#execution)
4. [Testing](#testing)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Abstract

### Problem Statement
The early detection and accurate diagnosis of leaf diseases in plants are crucial for agricultural practices and crop management. Accurate and timely identification of leaf diseases can help farmers take appropriate actions to mitigate the spread of diseases and improve crop yields. However, manual inspection of leaves for disease detection is a tedious and time-consuming task, and it requires expertise in plant pathology.

### Objective
The problem addressed is the development of an automated system that can accurately classify leaf diseases based on visual inspection. The system should be able to analyze leaf images, extract relevant features, and classify them into different disease categories or healthy leaves. The goal is to assist farmers and agricultural experts in early disease detection, enabling them to take preventive measures and improve crop health.

### Data Description
In this dataset, there are 39 different classes of plant leaf and background images. The classes include:

- Apple_scab
- Apple_black_rot
- Apple_cedar_apple_rust
- Apple_healthy
- Background_without_leaves
- Blueberry_healthy
- Cherry_powdery_mildew
- Cherry_healthy
- Corn_gray_leaf_spot
- Corn_common_rust
- Corn_northern_leaf_blight
- Corn_healthy
- Grape_black_rot
- Grape_black_measles
- Grape_leaf_blight
- Grape_healthy
- Orange_haunglongbing
- Peach_bacterial_spot
- Peach_healthy
- Pepper_bacterial_spot
- Pepper_healthy
- Potato_early_blight
- Potato_healthy
- Potato_late_blight
- Raspberry_healthy
- Soybean_healthy
- Squash_powdery_mildew
- Strawberry_healthy
- Strawberry_leaf_scorch
- Tomato_bacterial_spot
- Tomato_early_blight
- Tomato_healthy
- Tomato_late_blight
- Tomato_leaf_mold
- Tomato_septoria_leaf_spot
- Tomato_spider_mites_two-spotted_spider_mite
- Tomato_target_spot
- Tomato_mosaic_virus
- Tomato_yellow_leaf_curl_virus

The dataset is available [here](https://data.mendeley.com/datasets/tywbtsjrjv/1).

## DIP Components and Deep Learning Tools

1. **Image Preprocessing and Feature Extraction**:
   - Takes an image as input, resizes it to a specific size, and normalizes its pixel values.
   - Applies Canny edge detection to extract edge features from the preprocessed image.
   - Uses Local Binary Patterns (LBP) to extract texture features from the preprocessed image.
   - For each image in the training set, the code preprocesses the image, extracts edge and texture features, reshapes them, and concatenates them to form the final feature vector.
   - The extracted features are then converted to NumPy arrays and reshaped to match the expected input shape for the deep learning model.

2. **Model Architecture and Transfer Learning**:
   - The code uses the VGG16 model from Keras' applications module as the base model for transfer learning. VGG16 is a pre-trained convolutional neural network that has been trained on a large dataset (ImageNet) and can be fine-tuned for specific tasks.
   - The top layers of the VGG16 model are removed (include_top=False), and custom top layers are added for leaf disease detection. These layers include Global Average Pooling, Dense layers with ReLU activation, BatchNormalization, Dropout, and a final Dense layer with softmax activation for multi-class classification.

## Implementation

### Data Preparation
- The dataset directory is defined, and classes with their corresponding directories are specified.
- Images are loaded, resized, and converted to NumPy arrays.
- Image data is normalized, and the dataset is split into training, validation, and test sets.

### Image Preprocessing and Feature Extraction
- Images are preprocessed by resizing and normalizing pixel values.
- Edge features are extracted using Canny edge detection.
- Texture features are extracted using Local Binary Patterns (LBP).
- Edge and texture features are concatenated to form the final feature vector.

### Model Architecture and Transfer Learning
- The VGG16 model is used as the base for transfer learning, with its top layers removed.
- Custom top layers are added, including Global Average Pooling, Dense layers, BatchNormalization, Dropout, and a final Dense layer for multi-class classification.

### Model Compilation
- The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.

### Training
- The model is trained using the extracted features from the training set and their encoded labels.
- Early stopping and model checkpoint callbacks are used to monitor training and save the best model.
- Training history is stored in a DataFrame and saved as a CSV file.

## Testing

### FastAPI Plant Disease Detection API
A FastAPI application was built to predict plant diseases from uploaded leaf images. The model can classify various diseases and healthy conditions across multiple plant types.

### Features
- **Image Upload**: Upload an image of a leaf for disease prediction.
- **Prediction**: The API uses a pre-trained deep learning model to classify the disease.

### How It Works
1. **Image Upload**: Users upload an image of a leaf.
2. **Image Processing**: The image is resized and processed to match the input requirements of the model.
3. **Prediction**: The processed image is fed into the model, which returns the predicted class of the disease.
4. **Response**: The API returns the disease name from a predefined list of classes.

### Running the API

#### Prerequisites
- Python 3.7+
- Required Python packages: fastapi, pydantic, numpy, pillow, tensorflow, opencv-python, matplotlib, scikit-image, uvicorn

#### Setup
- Install dependencies:
  ```sh
  pip install fastapi pydantic numpy pillow tensorflow opencv-python matplotlib scikit-image uvicorn
  
- Ensure the model file `best_model.h5` is in the correct path.

#### Start the API
Run the following command to start the FastAPI server:
```sh
uvicorn app:app
```
This will start the server at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Testing the API
Using FastAPI Docs:
- Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your web browser.
- You'll see an interactive API documentation interface.
- Use the `/predict` endpoint to upload an image and get a prediction.

Demo FastAPI Video – [here](https://mahindraecolecentrale-my.sharepoint.com/:v:/g/personal/se23maid010_mahindrauniversity_edu_in/EXMQe8MR5vFDluNDJ-xj6_MBAXgk_20EKuY3NjGFm2RYRg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=MzZwBF).

## Results

- **Epoch (10) vs Training Accuracy & Validation Accuracy**

  Train Loss: 0.7492
  Train Accuracy: 0.7640

- **Test Accuracy**

  Test Loss: 0.7491933107376099
  Test Accuracy: 0.7640075087547302

![Epoch vs Accuracy](accuracy.png)

## Conclusion

- In this project, we developed an automated leaf disease classification system using digital image processing and deep learning techniques. We utilized the Plant Village dataset, which contains images of leaves with various diseases and healthy leaves. By applying image preprocessing, feature extraction, and transfer learning with the VGG16 model, we achieved accurate classification results. The system was evaluated using a separate testing dataset, and it demonstrated high accuracy in identifying different leaf diseases.

- The implementation of this project involved several key steps, including data preparation, data augmentation, image preprocessing, feature extraction, model architecture and transfer learning, model training, and evaluation. We utilized the FastAPI framework to create a web API endpoint for leaf disease prediction, making the system accessible to users.

- Through this project, we demonstrated the effectiveness of combining digital image processing and deep learning for leaf disease classification. The system can assist farmers, agricultural experts, and researchers in early disease detection, enabling timely interventions and improved crop management.
