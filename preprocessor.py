import json
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences

# Load the saved tokenizer from JSON
with open(os.path.join("models", "tokenizer.json"), "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(json.load(f))

def preprocess(text, tokenizer, max_length=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)
    return padded
