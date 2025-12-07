from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import whisper
import os
import pyttsx3
import tempfile
from ollama import AsyncClient
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import sqlite3
from fastapi import Form

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Allow frontend running on port 8080
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite database setup
DATABASE_URL = "sqlite:///./users.db"

def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            tokens INTEGER DEFAULT 100  -- Initial token balance in rupees equivalent
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Pricing structure
PRICING = {
    "text_input": 0.0008,  # ₹0.0008 per token
    "text_output": 0.0025,  # ₹0.0025 per token
    "speech_to_text": 0.008,  # ₹0.008 per second
    "text_to_speech": 0.001,  # ₹0.001 per character
}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Helper functions for authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if not user or not verify_password(password, user[2]):
        return False
    return {"id": user[0], "username": user[1], "tokens": user[3]}

def deduct_tokens(user_id: int, amount: float):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET tokens = tokens - ? WHERE id = ?", (amount, user_id))
    conn.commit()
    conn.close()

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Dummy token validation (in production, use JWT)
    username = token.split("-")[0]
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return {"id": user[0], "username": user[1], "tokens": user[3]}

# User registration
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    hashed_password = get_password_hash(password)
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    finally:
        conn.close()
    return {"message": "User registered successfully"}

# User login
@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # Dummy token generation
    token = f"{username}-dummy-token"
    return {"access_token": token, "token_type": "bearer"}

# Token balance endpoint
@app.get("/token-balance")
async def get_token_balance(current_user: dict = Depends(get_current_user)):
    return {"balance": current_user["tokens"]}

# Purchase tokens
@app.post("/purchase-tokens")
async def purchase_tokens(amount: int, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET tokens = tokens + ? WHERE id = ?", (amount, current_user["id"]))
    conn.commit()
    conn.close()
    return {"message": f"{amount} tokens purchased successfully"}

# Define Message model
class Message(BaseModel):
    role: str
    content: str

# Conversation history
conversation_history = []

# Load models
print("[INFO] Loading Whisper model...")
whisper_model = whisper.load_model("tiny")  # Use a larger model for better accuracy
print("[INFO] Whisper model loaded.")

print("[INFO] Initializing Ollama AsyncClient...")

# ollama_client = AsyncClient()
import os
ollama_client = AsyncClient(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

print("[INFO] Ollama AsyncClient ready.")

# Initialize pyttsx3 TTS engine
print("[INFO] Initializing pyttsx3 TTS engine...")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech rate
print("[INFO] pyttsx3 TTS engine ready.")


# Speech-to-text function
def speech_to_text(audio_file):
    """Transcribes audio to text using Whisper."""
    try:
        # Transcribe using Whisper
        result = whisper_model.transcribe(audio_file, language="en")  # Specify language for better accuracy
        transcription = result["text"]

        # Post-process the transcription
        cleaned_text = transcription
        return cleaned_text
    except Exception as e:
        print(f"[ERROR] Whisper transcription failed: {e}")
        return "Error in speech recognition."

# Text-to-speech function
def text_to_speech(text):
    """Converts text to speech using pyttsx3."""
    if not text or text.strip() == "":
        print("[ERROR] No text provided for TTS!")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_path = temp_audio.name
            
        tts_engine.save_to_file(text, temp_path)
        tts_engine.runAndWait()
        return temp_path
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Authenticate the WebSocket connection
    query_params = websocket.query_params
    token = query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return

    # Validate the token
    username = token.split("-")[0]
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        await websocket.close(code=1008, reason="Invalid token")
        return

    current_user = {"id": user[0], "username": user[1], "tokens": user[3]}

    if current_user["tokens"] <= 0:
        await websocket.close(code=1008, reason="Insufficient tokens")
        return

    await websocket.accept()
    print("[INFO] WebSocket connection established.")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            if not audio_bytes:
                await websocket.send_json({"error": "No audio data received."})
                continue

            # Save received audio as WAV
            audio_path = "received_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            print(f"[INFO] Audio received and saved as {audio_path}")

            # Step 1: Convert speech to text
            stt_cost = len(audio_bytes) / 16000 * PRICING["speech_to_text"]  # Assuming 16kHz audio
            if current_user["tokens"] < stt_cost:
                await websocket.close(code=1008, reason="Insufficient tokens for STT")
                return
            deduct_tokens(current_user["id"], stt_cost)

            # Fetch updated token balance
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT tokens FROM users WHERE id = ?", (current_user["id"],))
            updated_tokens = cursor.fetchone()[0]
            conn.close()

            # Send updated token balance to client
            await websocket.send_json({"tokens_remaining": updated_tokens})

            user_message_text = speech_to_text(audio_path)
            print(f"[INFO] Transcribed Text: {user_message_text}")
            await websocket.send_json({"text": user_message_text})

            # Step 2: Generate assistant response
            input_cost = len(user_message_text.split()) * PRICING["text_input"]
            assistant_response = await generate_response(user_message_text)
            output_cost = len(assistant_response.split()) * PRICING["text_output"]

            total_t2t_cost = input_cost + output_cost
            if current_user["tokens"] < total_t2t_cost:
                await websocket.close(code=1008, reason="Insufficient tokens for T2T")
                return
            deduct_tokens(current_user["id"], total_t2t_cost)

            # Fetch updated token balance
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT tokens FROM users WHERE id = ?", (current_user["id"],))
            updated_tokens = cursor.fetchone()[0]
            conn.close()

            # Send updated token balance to client
            await websocket.send_json({"tokens_remaining": updated_tokens})

            print(f"[INFO] AI Response: {assistant_response}")
            await websocket.send_json({"response": assistant_response})

            # Add to conversation history
            conversation_history.append(Message(role="user", content=user_message_text))
            conversation_history.append(Message(role="assistant", content=assistant_response))

            # Step 3: Convert response to speech
            tts_cost = len(assistant_response) * PRICING["text_to_speech"]
            if current_user["tokens"] < tts_cost:
                await websocket.close(code=1008, reason="Insufficient tokens for TTS")
                return
            deduct_tokens(current_user["id"], tts_cost)

            # Fetch updated token balance
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT tokens FROM users WHERE id = ?", (current_user["id"],))
            updated_tokens = cursor.fetchone()[0]
            conn.close()

            # Send updated token balance to client
            await websocket.send_json({"tokens_remaining": updated_tokens})

            audio_response_path = text_to_speech(assistant_response)

            if audio_response_path and os.path.exists(audio_response_path):
                with open(audio_response_path, "rb") as f:
                    audio_response_bytes = f.read()

                await websocket.send_bytes(audio_response_bytes)
                print("[INFO] Response audio sent to client.")
            else:
                print("[ERROR] TTS failed, sending error message.")
                await websocket.send_json({"error": "TTS conversion failed."})

    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        await websocket.close()

async def generate_response(user_input):
    """Generates a response using the Ollama AI model."""
    try:
        response = await ollama_client.chat(
            model="llama3.2:1b",
            messages=[{"role": msg.role, "content": msg.content} for msg in conversation_history] + [{"role": "user", "content": user_input}]
        )
        return response['message']['content'] if response and "message" in response else "I couldn't understand that."
    except Exception as e:
        print(f"[ERROR] Ollama API error: {e}")
        return "Sorry, I couldn't process your request."

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)