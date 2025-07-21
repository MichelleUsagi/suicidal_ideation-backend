
import logging
import csv
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model_and_tokenizer
from preprocessor import preprocess
import uvicorn

# Load model, tokenizer, and maxlen once at startup
model, tokenizer, maxlen = load_model_and_tokenizer()

# Initialize FastAPI app
app = FastAPI(title="Suicide Ideation Detection API")

# Setup structured logging
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

# In-memory prediction history (useful for testing or visualization)
history = []

# Input schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "✅ Suicide Ideation Detection API is up and running."}

@app.post("/predict")
def predict(input_data: TextInput):
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        # Preprocess input
        sequence = preprocess(text, tokenizer, maxlen)
        prediction = model.predict(sequence)[0][0]
        label = int(prediction >= 0.6)
        message = "High risk" if label == 1 else "Low risk"
        probability = float(prediction)
        timestamp = datetime.now().isoformat()

        result = {
            "timestamp": timestamp,
            "input_text": text,
            "prediction": label,
            "probability": probability,
            "message": message
        }

        # Log in memory and CSV
        history.append(result)
        with open(csv_log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, text, label, probability, message])

        logging.info(f"Prediction made: {result}")
        return result

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/history")
def get_history():
    return history[-20:]  # Return last 20 predictions

# Optional: Uvicorn entrypoint for local/dev Docker use
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
