import streamlit as st
import requests
from PIL import Image
import os
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(page_title="MindMate – Emotional Support App", page_icon="🧠", layout="wide")

# Load and center the logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=120)  # Adjust width as needed


# Title and description
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>MindMate</h1>
    <h4 style='text-align: center; color: #34495e;'>Your supportive companion for emotional check-ins</h4>
""", unsafe_allow_html=True)

# Tabs for chatbot, history, and feedback
menu = st.tabs(["💬 Chatbot", "📜 History", "📝 Feedback"])

# Tab 1 - Chatbot
with menu[0]:
    st.subheader("Chat with MindMate")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", "", key="chat_input")

    if st.button("Send"):
        if user_input.strip():
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
                if response.status_code == 200:
                    result = response.json()
                    label = "🚨 High Risk" if result['prediction'] == 1 else "✅ Low Risk"
                    reply = f"Prediction: {label}\nConfidence: {result['probability']:.2f}"
                    st.session_state.chat_history.append((user_input, reply))
                else:
                    st.session_state.chat_history.append((user_input, "Something went wrong."))
            except Exception as e:
                st.session_state.chat_history.append((user_input, f"Error: {e}"))

    for user_msg, bot_msg in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"<div style='margin-left:20px;color:#2c3e50;'>💡 <i>{bot_msg}</i></div>", unsafe_allow_html=True)

# Tab 2 - History
with menu[1]:
    st.subheader("Prediction History")
    csv_file = "prediction_logs.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        st.dataframe(df.tail(20))
    else:
        st.info("No predictions logged yet.")

# Tab 3 - Feedback
with menu[2]:
    st.subheader("We'd love your feedback 💬")
    name = st.text_input("Your Name (Optional):")
    comments = st.text_area("What can we improve or add?")
    if st.button("Submit Feedback"):
        if comments.strip():
            feedback_file = "feedback_log.csv"
            with open(feedback_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{name},{comments}\n")
            st.success("Thanks for your feedback!")
        else:
            st.warning("Please enter your comments before submitting.")
