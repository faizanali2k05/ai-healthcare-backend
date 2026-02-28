from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
import os
from dotenv import load_dotenv

# Hugging Face transformers for model inference
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# We no longer use Gemini; the backend now relies on a Hugging Face model

# Initialize Supabase
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Client Init Failed: {e}")

# Load the Hugging Face translation/generation model once at startup
try:
    tokenizer = AutoTokenizer.from_pretrained("FremyCompany/opus-mt-nl-en-healthcare")
    model = AutoModelForSeq2SeqLM.from_pretrained("FremyCompany/opus-mt-nl-en-healthcare")
except Exception as e:
    print(f"Failed to load HF model: {e}")
    tokenizer = None
    model = None

@app.route("/")
def home():
    return {"status": "Healthcare AI Backend is Live!"}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"reply": "Error: Invalid JSON or empty body."}), 200

        user_id = data.get("user_id")
        message = data.get("message")

        if not message:
            return jsonify({"reply": "Error: Send a proper message."}), 200

        # 1. Generate reply using the Hugging Face model
        try:
            if not model or not tokenizer:
                return jsonify({"reply": "Model not loaded on the server."}), 200

            inputs = tokenizer(message, return_tensors="pt")
            outputs = model.generate(**inputs)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as ge:
            return jsonify({"reply": f"Model inference error: {str(ge)}"}), 200

        # 2. Try Supabase
        if supabase and user_id:
            try:
                supabase.table("chat_history").insert({
                    "user_id": user_id,
                    "user_message": message,
                    "bot_reply": reply
                }).execute()
            except Exception as se:
                print(f"Database Save Failed: {se}")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Chat Route Critical Error: {str(e)}")
        return jsonify({"reply": f"Internal Server Error: {str(e)}"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
