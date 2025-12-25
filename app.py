import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types  # Required for GenerateContentConfig

app = Flask(__name__)

# --- Configuration ---
# API Key is pulled from the environment variable on Render [cite: 193]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the new Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Use gemma-3-4b-it for high speed and low latency in a "Lite" app [cite: 415]
MODEL_ID = 'gemma-3-4b-it' 

def filter_gemini_response(text):
    """
    Prevents AI self-references and handles common error phrases .
    """
    unauthorized_message = "I am not authorised to answer this question. My purpose is solely to refine your raw prompt."
    text_lower = text.lower()
    
    unauthorized_phrases = [
        "as a large language model", "i am an ai", "i was trained by", 
        "my training data", "i cannot fulfill this request"
    ]
    
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            return unauthorized_message
    return text

def ask_ai_for_prompt(user_input):
    """
    Unified logic for Gemma 3 using the GenerateContentConfig .
    """
    # Instructing Gemma 3 as a Professional Prompt Engineer [cite: 415]
    unified_prompt = (
        "Role: Professional Prompt Engineer.\n"
        "Task: Expand the user's short input into a detailed AI Image prompt.\n"
        "Constraint: Return ONLY the refined prompt text.\n\n"
        f"User Input: {user_input}"
    )

    # Applying the temperature and token limits you specified 
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.1
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=unified_prompt,
            config=config
        )
        
        raw_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_text).strip()
    
    except Exception as e:
        app.logger.error(f"AI Generation Error: {e}")
        return "An error occurred while generating your prompt. Please try again later."

@app.route('/')
def home():
    """Renders the main interface [cite: 541-542]."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handles the prompt generation request from the frontend [cite: 549-551]."""
    user_text = request.form.get('user_input', '').strip()
    
    if not user_text:
        return jsonify({"error": "Please enter some text."}), 400
    
    refined_result = ask_ai_for_prompt(user_text)
    return jsonify({"result": refined_result})

if __name__ == '__main__':
    # Standard Flask run configuration for local testing
    app.run(host='0.0.0.0', port=5000)
