import tensorflow as tf

# This function loads your model cleanly for deployment
def load_model_file():
    # Render / hosting cannot use local Windows ABSOLUTE paths
    # So keep the model file in the same folder as your .py files
    return tf.keras.models.load_model("model.h5")
