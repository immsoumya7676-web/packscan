import tensorflow as tf

model = tf.keras.models.load_model("model.keras")

print("Model loaded successfully!")
model.summary()