from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model("model_1.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = "uploads/" + file.filename
    file.save(filepath)

    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    
    # Assuming it's a classification model
    class_labels = ["Early Blight", "Late Blight","Healthy"]
    predicted_class = class_labels[np.argmax(prediction)]
    
    os.remove(filepath)  # Cleanup uploaded file
    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
