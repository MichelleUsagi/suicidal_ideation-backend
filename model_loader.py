import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def load_model_and_tokenizer():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models", "suicide_detector_model.h5")
    tokenizer_path = os.path.join(base_path, "models", "tokenizer.json")

    # Load model
   model = load_model(model_path, compile=False)

    # Load tokenizer from JSON
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

    # Ensure this matches your training config
    maxlen = 200

    return model, tokenizer, maxlen
