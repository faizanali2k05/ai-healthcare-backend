from flask import Flask, request, jsonify
from supabase import create_client
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION FROM RENDER ENVIRONMENT VARIABLES ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@app.route("/")
def home():
    return {"status": "Healthcare AI Backend is Live!"}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # 1. Get Response from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(message)
        reply = response.text if response.text else "AI could not generate a response."

        # 2. Save to Supabase (if configured)
        if supabase and user_id:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "user_message": message,
                "bot_reply": reply
            }).execute()

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
