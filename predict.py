import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Folders containing images
folders = [
    "datasets/original",
    "datasets/altered"
]

# Image size used during training
IMG_SIZE = (224, 224)

print("Model loaded successfully")
print("------------------------")

# Check every image in folders
for folder_path in folders:

    print("\nChecking:", folder_path)

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)

        # Ignore non-image files
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Load image
        img = image.load_img(
            img_path,
            target_size=IMG_SIZE
        )

        # Convert image to array
        img_array = image.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)

        confidence = prediction[0][0]

        if confidence > 0.5:
            result = "TAMPERED"
            confidence_value = confidence * 100
        else:
            result = "ORIGINAL"
            confidence_value = (1-confidence) * 100

        print("\nImage:", img_name)
        print("Result:", result)
        print("Confidence:", round(confidence_value, 2), "%")