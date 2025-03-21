import gradio as gr
from typing import List, Tuple
import google.generativeai as genai
import os
from dotenv import load_dotenv  # Import dotenv to read .env file

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY. Set it in the .env file.")

# Configure Google Gemini API Key
genai.configure(api_key=api_key)

# Initialize Gemini Model
model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use "gemini-pro" or "gemini-pro-vision" as needed

# Chatbot function
def chatbot(user_input: str, history: List[Tuple[str, str]]):
    messages = []

    # Convert chat history into Gemini message format
    for user_msg, ai_msg in history:
        messages.append({"role": "user", "parts": [user_msg]})  # User message
        messages.append({"role": "model", "parts": [ai_msg]})   # Assistant response

    messages.append({"role": "user", "parts": [user_input]})  # Add new user message
    
    # Get response from Gemini API
    response = model.generate_content(messages)

    return response.text  # Extract response text

# Build Gradio Chat UI
iface = gr.ChatInterface(
    fn=chatbot,
    title="Google Gemini AI Chatbot",
    description="Chatbot powered by Google Gemini AI",
    theme="soft"
)

# Launch the app
iface.launch()
