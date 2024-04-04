from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

# Define the TextVectorization layer
max_features = 10000
sequence_length = 30
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Load the TensorFlow SavedModel
model = tf.keras.models.load_model(r'C:\Users\sneas\Downloads\Settyl\my_model.h5', compile=False)

# Adapt the vectorize_layer with some sample data
sample_data = ["Sample text data for adaptation"]
vectorize_layer.adapt(sample_data)

# Define the labels
labels = [
    "Arrival",
    "Departure",
    "Empty Container Released",
    "Empty Return",
    "Gate In",
    "Gate Out",
    "In-transit",
    "Inbound Terminal",
    "Loaded on Vessel",
    "Off Rail",
    "On Rail",
    "Outbound Terminal",
    "Port In",
    "Port Out",
    "Unloaded on Vessel"
]

# Initialize and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

class Status(BaseModel):
    external_status: str

app = FastAPI()

@app.post("/predict/")
def predict_status(status: Status):
    # Preprocess the input
    vectorized_status = vectorize_layer([status.external_status])
    
    # Convert the input tensor to float type
    vectorized_status = tf.cast(vectorized_status, tf.float32)
    
    # Make a prediction using the model
    prediction = model.predict(vectorized_status)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    return {"internal_status": predicted_label}