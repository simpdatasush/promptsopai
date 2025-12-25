import os
from flask import Flask, render_template, request, jsonify
from google import genai

app = Flask(__name__)

# --- Configuration ---
# API Key is pulled from the environment variable we will set in Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def ask_ai_for_prompt(user_input):
    """Unified logic for Gemma 3 or Gemini 2.0."""
    unified_prompt = (
        "Role: Professional Prompt Engineer.\n"
        "Task: Expand the user's short input into a detailed AI Image prompt.\n"
        "Constraint: Return ONLY the refined prompt text.\n\n"
        f"User Input: {user_input}"
    )
    try:
        # Using gemma-3-4b-it for high speed on lite apps
        response = client.models.generate_content(
            model='gemma-3-4b-it', 
            contents=unified_prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_text = request.form.get('user_input', '').strip()
    if not user_text:
        return jsonify({"error": "Please enter some text."}), 400
    
    refined_result = ask_ai_for_prompt(user_text)
    return jsonify({"result": refined_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
