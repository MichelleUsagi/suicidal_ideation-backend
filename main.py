
import logging
import csv
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model_and_tokenizer
from preprocessor import preprocess

# Load model, tokenizer, and maxlen
model, tokenizer, maxlen = load_model_and_tokenizer()

# Initialize FastAPI app
app = FastAPI()

# Setup structured logging to a file
logging.basicConfig(
    filename="prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setup CSV logging
csv_log_file = "prediction_logs.csv"
if not os.path.exists(csv_log_file):
    with open(csv_log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "input_text", "prediction", "probability", "message"])

# In-memory browser history
history = []

# Request body schema
class TextInput(BaseModel):
    text: str

# Health check endpoint
@app.get("/")
def home():
    return {"message": "Suicide Ideation Detection API running."}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        # Preprocess input and get prediction
        sequence = preprocess(input_data.text, tokenizer, maxlen)
        prediction = model.predict(sequence)[0][0]
        label = int(prediction >= 0.6)
        message = "High risk" if label == 1 else "Low risk"
        probability = float(prediction)
        timestamp = datetime.now().isoformat()

        # Prepare result
        result = {
            "timestamp": timestamp,
            "input_text": input_data.text,
            "prediction": label,
            "probability": probability,
            "message": message
        }

        # Save to in-memory history and CSV
        history.append(result)
        with open(csv_log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, input_data.text, label, probability, message])

        # Log the prediction
        logging.info(f"Prediction: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

# Endpoint to retrieve recent predictions
@app.get("/history")
def get_history():
    return history[-20:]
