from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
import numpy as np
import uuid
import tensorflow as tf
import json
import os
import requests
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

# ---------------- AI CHATBOT ROUTE ----------------
@app.route('/ai-chat', methods=['POST'])
def ai_chat():
    try:
        # Get the message from JSON
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"reply": "No message received."})

        # OpenRouter API request
        headers = {
            "Authorization": "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",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [{"role": "user", "content": user_message}]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )

        # Parse JSON safely
        result = response.json()
        # Debug: print the full response to console
        print("OpenRouter Response:", json.dumps(result, indent=2))

        # Safely get the reply
        reply = "Sorry, I couldn't get a response."
        choices = result.get("choices")
        if choices and len(choices) > 0:
            message_obj = choices[0].get("message")
            if message_obj:
                reply = message_obj.get("content", reply)

        return jsonify({"reply": reply})

    except Exception as e:
        print("Chatbot error:", e)
        return jsonify({"reply": "Sorry, I couldn't process your request."})
if __name__ == "__main__":
    app.run(debug=True)