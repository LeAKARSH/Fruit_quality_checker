import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Configuration
DATA_DIR = 'data/training'  # Updated path to training data
IMG_SIZE = 50
MODEL_PATH = 'fruitquality-structured-cnn.h5'

# Get fruit types
FRUIT_TYPES = []
for item in os.listdir(DATA_DIR):
    fruit_path = os.path.join(DATA_DIR, item)
    if os.path.isdir(fruit_path):
        FRUIT_TYPES.append(item)
print(f"Found fruit types: {FRUIT_TYPES}")

# Create categories and label mapping
CATEGORIES = []
label_map = {}
index = 0
for fruit in FRUIT_TYPES:
    for quality in ['fresh', 'rotten']:
        cat = f"{quality}_{fruit}"
        CATEGORIES.append(cat)
        label_map[cat] = index
        index += 1
print(f"Categories: {CATEGORIES}")

def load_test_data():
    test_data = []
    y_true = []

    for fruit in FRUIT_TYPES:
        fruit_path = os.path.join(DATA_DIR, fruit)
        if not os.path.exists(fruit_path):
            print(f"Warning: Path does not exist: {fruit_path}")
            continue

        for quality in ['fresh', 'rotten']:
            label_key = f"{quality}_{fruit}"
            if label_key not in label_map:
                print(f"Warning: {label_key} not found in label_map.")
                continue

            label = label_map[label_key]
            path = os.path.join(fruit_path, quality)
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                continue

            for img in os.listdir(path):
                try:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img_array is None:
                        print(f"Skipped unreadable image: {img_path}")
                        continue

                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    test_data.append(img_array)
                    y_true.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    if not test_data:
        raise ValueError("No valid test images found. Please check your dataset path and image files.")

    return np.array(test_data), np.array(y_true)

try:
    # Load test data
    X_test, y_test = load_test_data()
    print(f"Loaded {len(X_test)} test images")

    # Preprocess
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test / 255.0  # Normalize
    
    # Load and evaluate model
    model = load_model(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
except Exception as e:
    print(f"Error: {e}") 