import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def load_model_and_tokenizer():
    # Get absolute base path where this script is located
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Build full paths to model and tokenizer
    model_path = os.path.join(base_path, "models", "suicide_detector_model.h5")
    tokenizer_path = os.path.join(base_path, "models", "tokenizer.json")

    # DEBUG: Print paths and check file existence
    print("Loading model from:", model_path)
    print("Model file exists:", os.path.exists(model_path))
    print("Loading tokenizer from:", tokenizer_path)
    print("Tokenizer file exists:", os.path.exists(tokenizer_path))

    # Load model without optimizer (for deployment safety)
    model = load_model(model_path, compile=False)

    # Load tokenizer
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

    # Set maxlen used during training
    maxlen = 200

    return model, tokenizer, maxlen
