import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def load_model_and_tokenizer():
    """
    Load the trained suicide detection model and tokenizer from the 'models/' directory.
    Ensures absolute paths and prints debug info for deployment validation.
    """
    # Absolute path to the directory this file is in
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Build full paths
    model_path = os.path.join(base_path, "models", "suicide_detector_model.h5")
    tokenizer_path = os.path.join(base_path, "models", "tokenizer.json")

    # Debug prints to confirm paths
    print("Loading model from:", model_path)
    print("Model exists:", os.path.exists(model_path))
    print("Loading tokenizer from:", tokenizer_path)
    print("Tokenizer exists:", os.path.exists(tokenizer_path))

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Model or tokenizer file not found in models/ folder.")

    # Load model without compiling (for inference-only use)
    model = load_model(model_path, compile=False)

    # Load tokenizer from JSON
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

    # Max sequence length used during training
    maxlen = 200

    return model, tokenizer, maxlen
