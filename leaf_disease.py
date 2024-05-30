import os
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from skimage import exposure, feature
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Data Preparation

# Download the dataset and extract images
dataset_dir = 'data'

# Define classes and their corresponding directories
classes = ['Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy', 'Background_without_leaves', 'Blueberry_healthy',
           'Cherry_powdery_mildew', 'Cherry_healthy', 'Corn_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy',
           'Grape_black_rot', 'Grape_black_measles', 'Grape_leaf_blight', 'Grape_healthy', 'Orange_haunglongbing', 'Peach_bacterial_spot',
           'Peach_healthy', 'Pepper_bacterial_spot', 'Pepper_healthy', 'Potato_early_blight', 'Potato_healthy', 'Potato_late_blight',
           'Raspberry_healthy', 'Soybean_healthy', 'Squash_powdery_mildew', 'Strawberry_healthy', 'Strawberry_leaf_scorch', 'Tomato_bacterial_spot',
           'Tomato_early_blight', 'Tomato_healthy', 'Tomato_late_blight', 'Tomato_leaf_mold', 'Tomato_septoria_leaf_spot',
           'Tomato_spider_mites_two-spotted_spider_mite', 'Tomato_target_spot', 'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus']

class_dirs = {cls: os.path.join(dataset_dir, cls) for cls in classes}

# Resize and normalize images, and split into training, validation, and test sets
image_size = (224, 224)
data = []
labels = []

for cls, cls_dir in class_dirs.items():
    for filename in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, filename)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        data.append(img_array)
        labels.append(cls)


data = np.array(data) / 255.0
labels = np.array(labels)


# Split the data into training, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)


# Step 2: Image Preprocessing and Feature Extraction

# Resize, normalize, and convert images to grayscale
def preprocess_image(img):
    # img = cv2.resize(img, image_size)/
    img = img / 255.0

    # Convert the image to grayscale if it's not already
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

# Extract edge features using Canny edge detection
def extract_edge_features(img):
    # Convert image to grayscale if necessary
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Check image depth and resize if necessary
    if img.dtype != np.uint8 or img.shape[0] % 2 != 0 or img.shape[1] % 2 != 0:
        img = img.astype(np.uint8)
        img = cv2.resize(img, (img.shape[1], img.shape[0]))

    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)

    return edges

# Extract texture features using local binary patterns
def extract_texture_features(img):
    lbp = feature.local_binary_pattern(img, 8, 1)
    return lbp


# Preprocess images and extract features from training data
train_features = []
for img in train_data:
    preprocessed_img = preprocess_image(img)
    edge_features = extract_edge_features(preprocessed_img)

    # edge_features_reshaped = edge_features.reshape(edge_features.shape[0], edge_features.shape[1], 1)

    texture_features = extract_texture_features(preprocessed_img)

    # texture_features_reshaped = texture_features.reshape(texture_features.shape[0], texture_features.shape[1], 1)

     # Reshape edge features to have the same number of dimensions as texture features
    edge_features_reshaped = edge_features.reshape(1, edge_features.shape[0], edge_features.shape[1])

    # Reshape texture features to have the same number of dimensions as edge features
    texture_features_reshaped = texture_features.reshape(1, texture_features.shape[0], texture_features.shape[1])

    # print("Edge features reshaped shape:", edge_features_reshaped.shape)
    # print("Texture features reshaped shape:", texture_features_reshaped.shape)
    # Concatenate edge and texture features along a new axis
    combined_features = np.concatenate((edge_features_reshaped, texture_features_reshaped), axis=0)

    # Take the mean along the new axis to collapse it
    combined_features = np.mean(combined_features, axis=0)
    train_features.append(combined_features)

train_features = np.array(train_features)

# Encode class labels into numerical values
encoder = LabelEncoder()
train_labels_encoded = encoder.fit_transform(train_labels)
val_labels_encoded = encoder.transform(val_labels)
test_labels_encoded = encoder.transform(test_labels)

# Extract features from validation images
val_features = []
for img in val_data:
    preprocessed_img = preprocess_image(img)
    edge_features = extract_edge_features(preprocessed_img)
    texture_features = extract_texture_features(preprocessed_img)
    edge_features_reshaped = edge_features.reshape(1, edge_features.shape[0], edge_features.shape[1])
    texture_features_reshaped = texture_features.reshape(1, texture_features.shape[0], texture_features.shape[1])
    combined_features = np.concatenate((edge_features_reshaped, texture_features_reshaped), axis=0)
    combined_features = np.mean(combined_features, axis=0)
    val_features.append(combined_features)

val_features = np.array(val_features)

# Reshape extracted features to match the expected input shape
train_features = np.expand_dims(train_features, axis=-1)  # Add a channel dimension if the images are grayscale
train_features = np.tile(train_features, (1, 1, 1, 3))  # Convert grayscale to RGB if needed

val_features = np.expand_dims(val_features, axis=-1)  # Add a channel dimension if the images are grayscale
val_features = np.tile(val_features, (1, 1, 1, 3))  # Convert grayscale to RGB if needed

# train_features_reshaped = train_features.reshape(-1, image_size[0], image_size[1], 3)
# val_features_reshaped = val_features.reshape(-1, image_size[0], image_size[1], 3)

# Step 3: Model Architecture and Transfer Learning

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_size + (3,))

# Add custom top layers for leaf disease detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Training

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)


# Train the model using extracted features
history = model.fit(train_features, to_categorical(train_labels_encoded),
                  validation_data=(val_features, to_categorical(val_labels_encoded)),
                  epochs=10,
                  callbacks=[early_stopping, checkpoint])
# Convert the history.history dictionary to a DataFrame
history_df = pd.DataFrame(history.history)

# Save the DataFrame to a CSV file
history_df.to_csv('training_history.csv', index=False)


# Step 5: Evaluation

# Evaluate the model on the test set using extracted features
test_features = []
for img in test_data:
    preprocessed_img = preprocess_image(img)
    edge_features = extract_edge_features(preprocessed_img)
    texture_features = extract_texture_features(preprocessed_img)
    edge_features_reshaped = edge_features.reshape(1, edge_features.shape[0], edge_features.shape[1])
    texture_features_reshaped = texture_features.reshape(1, texture_features.shape[0], texture_features.shape[1])
    combined_features = np.concatenate((edge_features_reshaped, texture_features_reshaped), axis=0)
    combined_features = np.mean(combined_features, axis=0)
    test_features.append(combined_features)

test_features = np.array(test_features)

test_features = np.expand_dims(test_features, axis=-1)  # Add a channel dimension if the images are grayscale
test_features = np.tile(test_features, (1, 1, 1, 3))  # Convert grayscale to RGB if needed

loss, accuracy = model.evaluate(test_features, to_categorical(test_labels_encoded))
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
