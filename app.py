from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import uuid
import tensorflow as tf
import json
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("models/plant_disease_corn_mobilenet.keras")

# Only Corn classes
labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


# Load disease info JSON
with open("plant_disease.json", "r") as f:
    plant_disease = json.load(f)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Preprocess image using EfficientNet preprocessing
def extract_features(image):
    img = tf.keras.utils.load_img(image, target_size=(160, 160))  # <-- must match your trained model
    feature = tf.keras.utils.img_to_array(img)
    feature = np.expand_dims(feature, axis=0)
    feature = feature / 255.0
    return feature

# Predict function
def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)[0]
    index = np.argmax(prediction)
    confidence = prediction[index] * 100

    # Safety check
    if index >= len(plant_disease):
        return {
            "name": "Unknown",
            "cause": "Prediction index out of range",
            "cure": "Try another image",
            "confidence": "0%"
        }

    # Return dictionary including confidence
    result = plant_disease[index]
    result["confidence"] = f"{confidence:.2f}%"
    return result

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        filepath = os.path.join("uploadimages", filename)

        # Ensure folder exists
        os.makedirs("uploadimages", exist_ok=True)
        image.save(filepath)

        prediction = model_predict(filepath)

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/uploadimages/{filename}',
            prediction=prediction
        )
    else:
        return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)