import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# --- Configuration ---
IMG_SIZE = 50
UPLOAD_FOLDER = 'C:/Users/Akarshjayanth.M.Naik/Desktop/my_fruit_checker/static/uploads'
MODEL_PATH = 'C:/Users/Akarshjayanth.M.Naik/Desktop/my_fruit_checker/fruitquality-structured-cnn.h5'
# Match with your model's category order
CATEGORIES = ['fresh_apple', 'rotten_apple', 'fresh_banana', 'rotten_banana', 'fresh_orange', 'rotten_orange']

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model(MODEL_PATH)

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            filename = file.filename
    return render_template('index.html', prediction=prediction, filename=filename)

def predict_image(image_path):
    try:
        img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)[0]  # Single image
        class_index = np.argmax(prediction)
        confidence = prediction[class_index] * 100

        predicted_label = CATEGORIES[class_index]
        quality = 'Good (Fresh)' if 'fresh' in predicted_label else 'Bad (Rotten)'

        return {
            'label': predicted_label,
            'quality': quality,
            'confidence': f"{confidence:.2f}%"
        }
    except Exception as e:
        return {'error': str(e)}


# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
