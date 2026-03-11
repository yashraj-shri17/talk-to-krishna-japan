"""
Flask API server for Talk to Krishna web interface.
This provides a REST API endpoint for the web frontend.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import io

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gita_api import GitaAPI
from src.config import settings
import edge_tts
import asyncio
import uuid
from flask import send_file

# Create audio cache directory
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio_cache')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# In-memory audio cache for fast serving
audio_cache = {}
import threading

async def _generate_multi_voice_audio(text: str, buffer: io.BytesIO, language: str = 'japanese'):
    """
    Detect language parts and generate audio using multiple voices sequentially.
    """
    import re
    # Clean text but keep structure for language detection
    clean_text = re.sub(r'<[^>]*>', '', text)
    
    lines = clean_text.split('\n')
    segments = []
    
    # Force default voice to be Japanese as per new requirements
    default_voice = "ja-JP-KeitaNeural"
    current_voice = default_voice
    current_text_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_text_lines:
                current_text_lines.append(line)
            continue
            
        # Any Devanagari character triggers Hindi voice for the line
        has_devanagari = any('\u0900' <= char <= '\u097F' for char in stripped)
        target_voice = "hi-IN-MadhurNeural" if has_devanagari else default_voice
        
        if target_voice != current_voice:
            if current_text_lines:
                segments.append((current_voice, "\n".join(current_text_lines)))
            current_text_lines = [line]
            current_voice = target_voice
        else:
            current_text_lines.append(line)
            
    if current_text_lines:
        segments.append((current_voice, "\n".join(current_text_lines)))
        
    # Always isolate "Bhagwat geeta Chapter X Shloka Y" (and variations) for English pronunciation
    refined_segments = []
    for voice, segment_text in segments:
        if not segment_text.strip():
            continue
            
        # Split text, capturing variations like "Bhagwat geeta Chapter X Shloka Y" or just "Chapter X Shloka Y"
        parts = re.split(r'(?i)((?:Bhagwat\s*geeta[\s,]*)?Chapter\s+\d+(?:[\s,]*Shloka\s+\d+)?)', segment_text)
        
        for part in parts:
            if not part.strip():
                if part:
                    refined_segments.append((voice, part))
                continue
                
            # If this part is the English chapter/shloka reference, use English voice
            if re.match(r'(?i)^(?:Bhagwat\s*geeta[\s,]*)?Chapter\s+\d+', part.strip()):
                refined_segments.append(("en-IN-PrabhatNeural", part))
            else:
                refined_segments.append((voice, part))

    # Generate audio for each segment and append to buffer
    for voice, segment_text in refined_segments:
        if not segment_text.strip():
            continue
            
        # Slightly faster rate for Japanese, normal for Sanskrit/Hindi/English
        rate = "+10%" if "ja-JP" in voice else "+0%"
        
        clean_segment = segment_text[:40].replace('\n', ' ')
        print(f"TTS Segment [{voice}]: {clean_segment}...")
        communicate = edge_tts.Communicate(segment_text, voice, rate=rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

def _generate_audio_async(text: str, language: str = 'japanese') -> str:
    """
    Generate audio asynchronously and cache it.
    Returns audio_id immediately while generation happens in background.
    """
    audio_id = str(uuid.uuid4())
    
    def generate():
        try:
            import time
            gen_start = time.time()
            print(f"Starting TTS generation for audio_id: {audio_id}")
            
            # Clean text
            import re
            clean_text = re.sub(r'<[^>]*>', '', text).replace('\n', ' ')
            print(f"Text length: {len(clean_text)} characters")
            
            # Buffer to hold audio in memory
            audio_buffer = io.BytesIO()
            
            # Run async generation with multi-voice support
            tts_start = time.time()
            asyncio.run(_generate_multi_voice_audio(text, audio_buffer, language))
            tts_time = time.time() - tts_start
            
            # Reset buffer pointer
            audio_buffer.seek(0)
            
            # Cache the audio data
            audio_cache[audio_id] = audio_buffer.getvalue()
            
            total_time = time.time() - gen_start
            audio_size = len(audio_cache[audio_id]) / 1024
            print(f"TTS complete: {tts_time:.2f}s, Total: {total_time:.2f}s, Size: {audio_size:.1f}KB")
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            audio_cache[audio_id] = None
    
    # Start generation in background thread
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()
    
    return audio_id


app = Flask(__name__)

# CORS configuration
from dotenv import load_dotenv
import re

load_dotenv()

frontend_url = os.getenv('FRONTEND_URL')
# Default local development origins
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if frontend_url:
    normalized_url = frontend_url.rstrip('/')
    allowed_origins.extend([normalized_url, f"{normalized_url}/"])

CORS(app, origins=allowed_origins, supports_credentials=True)

@app.before_request
def log_request_info():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {request.method} {request.path} from {request.remote_addr}")
    if request.headers.get('Origin'):
        print(f"  Origin: {request.headers.get('Origin')}")


# Initialize GitaAPI once
print("Initializing Talk to Krishna API...")
gita_api = GitaAPI()
gita_api._load_resources()
print("API Ready!\n")

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Handle question from web interface.
    
    Request JSON:
        {
            "question": "user's question here",
            "include_audio": true/false (optional, default: false),
            "user_id": 123 (optional, for logged-in users)
        }
    
    Response JSON:
        {
            "answer": "Krishna's response",
            "shlokas": [...],
            "audio_url": "/api/audio/<id>" (if include_audio=true),
            "success": true
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'No question provided',
                'success': False
            }), 400
        
        question = data['question'].strip()
        include_audio = data.get('include_audio', False)
        user_id = data.get('user_id')  # Optional: for logged-in users
        
        session_id = data.get('session_id')  # New: Session ID for context filtering
        language = data.get('language', 'japanese') # 'japanese' or 'english'
        
        # Check chat access for logged-in users
        if user_id:
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute('SELECT has_chat_access, role FROM users WHERE id = %s', (user_id,))
                user_access = c.fetchone()
                conn.close()
                
                if user_access:
                    has_access, role = user_access
                    if not has_access and role != 'admin':
                        return jsonify({
                            'error': 'You do not have permission to use the chat feature. Please contact the administrator.',
                            'success': False,
                            'access_denied': True
                        }), 403
            except Exception as e:
                print(f"Error checking chat access: {e}")
                # Log the error and continue - don't block user if DB check fails
        
        if not question:
            return jsonify({
                'error': 'Question cannot be empty',
                'success': False
            }), 400

        # --- FAST GREETING CHECK (Backup) ---
        # This ensures we catch greetings at the API layer 
        # to guarantee instant response without DB lookup.
        greetings_backup = {
            # English greetings
            "hi", "hello", "hey", "hii", "hiii", "helo", "heyy", "heya", "yo",
            "greetings", "good morning", "good afternoon", "good evening", "good night",
            "gm", "ge", "gn", "ga", "morning", "evening", "afternoon",
            
            # Hindi/Sanskrit greetings (Romanized)
            "namaste", "namaskar", "namaskaram", "pranam", "pranaam", "pranaams",
            "radhe radhe", "radhey radhey", "radhe", "radhey",
            "jai shri krishna", "jai shree krishna", "jai sri krishna", 
            "hare krishna", "hare krsna", "krishna", "krsna",
            "jai", "jay", "om", "aum",
            
            # Hindi Devanagari Script Greetings
            "हेलो", "हेल्लो", "हाय", "हाई", "हलो",
            "नमस्ते", "नमस्कार", "नमस्कारम", "प्रणाम", "प्रनाम",
            "राधे राधे", "राधे", "राधेय राधेय",
            "जय श्री कृष्ण", "जय श्रीकृष्ण", "जय कृष्ण",
            "हरे कृष्ण", "हरे कृष्णा", "कृष्ण",
            "जय", "ओम", "ॐ",
            "सुप्रभात", "शुभ संध्या", "शुभ रात्रि",
            "कैसे हो", "कैसे हैं", "क्या हाल", "क्या हाल है",
            
            # Casual/Informal
            "sup", "wassup", "whatsup", "howdy", "hola",
            "kaise ho", "kaise hain", "kya haal", "kya hal", "namaskaar"
        }
        
        import unicodedata
        q_lower = "".join(c for c in question.lower() if c.isalnum() or c.isspace() or unicodedata.category(c).startswith('M'))
        q_words = q_lower.split()
        
        is_greeting = False
        if q_words:
            # Check if entire query is a greeting phrase
            full_query = ' '.join(q_words)
            if full_query in greetings_backup:
                is_greeting = True
            
            # Check for two-word greeting phrases
            elif len(q_words) >= 2:
                two_word = f"{q_words[0]} {q_words[1]}"
                if two_word in greetings_backup:
                    if len(q_words) <= 3:
                        is_greeting = True
                    else:
                        q_words_set = {'what', 'how', 'why', 'who', 'when', 'where', 
                                     'kya', 'kyun', 'kaise', 'kab', 'kahan', 'kaun',
                                     'explain', 'tell', 'batao', 'bataiye', 'btao'}
                        if not any(qw in q_words for qw in q_words_set):
                            is_greeting = True
            
            # Case 1: Very short (just greeting)
            elif len(q_words) <= 3 and any(w in greetings_backup for w in q_words):
                is_greeting = True
                
            # Case 2: Greeting start, no question words
            elif len(q_words) <= 6 and q_words[0] in greetings_backup:
                q_words_set = {'what', 'how', 'why', 'who', 'when', 'where', 
                             'kya', 'kyun', 'kaise', 'kab', 'kahan', 'kaun',
                             'explain', 'tell', 'batao', 'bataiye', 'btao',
                             'is', 'are', 'can', 'should', 'would', 'could'}
                if not any(qw in q_words for qw in q_words_set):
                    is_greeting = True



        if is_greeting:
            print(f"Greeting detected in API: {question}")
            # Always return greeting in Japanese as per user request
            greeting_text = "ラーデー・ラーデー！私はシュリー・クリシュナです。何かお手伝いできることはありますか？"
            response = {
                'success': True,
                'answer': greeting_text,
                'shlokas': [],
                'llm_used': True 
            }
            
            # Save greeting conversation if user is logged in
            if user_id:
                save_conversation(user_id, question, greeting_text, [], session_id=session_id)
            
            # Generate audio if requested
            if include_audio:
                audio_id = _generate_audio_async(greeting_text, language)
                response['audio_url'] = f'/api/audio/{audio_id}'
                print(f"Greeting audio generated: {audio_id}")
            
            return jsonify(response)
        # ------------------------------------
        
        # Get user's conversation history if logged in
        conversation_history = []
        if user_id:
            # Filter history by session_id if provided
            conversation_history = get_user_history(user_id, session_id=session_id, limit=5)
            print(f"Retrieved {len(conversation_history)} previous conversations for user {user_id} (Session: {session_id})")
        
        # Get answer from GitaAPI with NO conversation context (1 Q = 1 ans requested by user)
        import time
        start_time = time.time()
        result = gita_api.search_with_llm(question, conversation_history=[], language=language)
        llm_time = time.time() - start_time
        
        answer_text = result.get('answer')
        all_shlokas = result.get('shlokas', [])
        chosen_shloka_id = result.get('chosen_shloka_id')
        
        # Only keep the shloka that was ACTUALLY chosen and spoken by the LLM
        shlokas_to_save = []
        if chosen_shloka_id:
            shlokas_to_save = [s for s in all_shlokas if s['id'] == chosen_shloka_id]
            # If for some reason the LLM quoted one not in the retrieved top 5, just store the ID
            if not shlokas_to_save:
                shlokas_to_save = [{'id': chosen_shloka_id}]
        else:
            # Fallback if regex failed to extract
            shlokas_to_save = all_shlokas[:1] if all_shlokas else []
        
        # Save conversation if user is logged in
        if user_id and answer_text:
            save_conversation(user_id, question, answer_text, shlokas_to_save, session_id=session_id)
            print(f"Saved conversation for user {user_id} with shloka {shlokas_to_save[0]['id'] if shlokas_to_save else 'None'}")
        
        # Format response (Can still return all to UI if needed, or just the chosen one. Returning only chosen one for consistency)
        response = {
            'success': True,
            'answer': answer_text,
            'shlokas': shlokas_to_save,
            'llm_used': result.get('llm_used', False)
        }
        
        # Generate audio in parallel if requested
        if include_audio and answer_text:
            audio_start = time.time()
            audio_id = _generate_audio_async(answer_text, language)
            audio_time = time.time() - audio_start
            response['audio_url'] = f'/api/audio/{audio_id}'
            print(f"Timing: LLM={llm_time:.2f}s, Audio={audio_time:.2f}s")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/speak', methods=['POST'])
def speak_text():
    """
    Generate audio from text using Neural TTS in-memory (no files saved).
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'japanese')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Buffer to hold audio in memory
        audio_buffer = io.BytesIO()
        
        # Run async generation with multi-voice support
        asyncio.run(_generate_multi_voice_audio(text, audio_buffer, language))

        # Reset buffer pointer to beginning
        audio_buffer.seek(0)

        # Return file from memory
        return send_file(
            audio_buffer,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="response.mp3"
        )

    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio using Groq Whisper-large-v3.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided', 'success': False}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'japanese')
        
        # Save temp file
        temp_path = os.path.join(AUDIO_DIR, f"temp_{uuid.uuid4()}.webm")
        audio_file.save(temp_path)
        
        # Call Groq
        with open(temp_path, "rb") as file:
            prompt_str = "The user is speaking in Japanese, Hindi, or English. Please transcribe Japanese in Kanji/Kana and Hindi in Devanagari script. ラーデー・ラーデー、お元気ですか？ नमस्ते, आप कैसे हैं? Hello, how are you?"
            if language == 'english':
                prompt_str = "The user is speaking in English. Please transcribe in English. Radhe Radhe, how are you?"
                
            transcription = gita_api.groq_client.audio.transcriptions.create(
                file=(audio_file.filename, file.read()),
                model="whisper-large-v3",
                prompt=prompt_str,
            )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        print(f"Transcribed Text: {transcription.text}")
        return jsonify({'text': transcription.text, 'success': True})
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/audio/<audio_id>', methods=['GET'])
def get_audio(audio_id):
    """
    Serve pre-generated audio from cache.
    Polls for audio to be ready if still generating.
    """
    import time
    max_wait = 20  # Increased to 20 seconds for Edge TTS
    start_time = time.time()
    
    print(f"Audio request for ID: {audio_id}")
    
    while time.time() - start_time < max_wait:
        if audio_id in audio_cache:
            audio_data = audio_cache[audio_id]
            
            if audio_data is None:
                print(f"Audio generation failed for {audio_id}")
                return jsonify({'error': 'Audio generation failed'}), 500
            
            elapsed = time.time() - start_time
            print(f"Audio ready after {elapsed:.2f}s")
            
            # Serve from memory
            import io
            audio_buffer = io.BytesIO(audio_data)
            audio_buffer.seek(0)
            
            return send_file(
                audio_buffer,
                mimetype="audio/mpeg",
                as_attachment=False,
                download_name="response.mp3"
            )
        
        # Wait a bit before checking again
        time.sleep(0.1)
    
    elapsed = time.time() - start_time
    print(f"Audio timeout after {elapsed:.2f}s for {audio_id}")
    return jsonify({'error': 'Audio not ready yet', 'waited': f'{elapsed:.2f}s'}), 404


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Talk to Krishna API',
        'version': '2.0.0'
    })

@app.route('/')
def index():
    """Serve basic info."""
    return jsonify({
        'message': 'Talk to Krishna API',
        'endpoints': {
            '/api/ask': 'POST - Ask a question',
            '/api/health': 'GET - Health check'
        }
    })

import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime
import re
from collections import defaultdict
import time

# Database setup
# Allow overriding db path for production environments (like Render persistent disks)
import os
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://neondb_owner:npg_AIJCOKgs6hN4@ep-twilight-field-ail5wonj-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

def get_db_connection():
    max_retries = 5
    base_delay = 0.5
    
    start_time = time.time()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                print(f"  DB Connect took {elapsed:.2f}s")
            return conn
        except psycopg2.OperationalError as e:
            last_error = e
            if attempt == max_retries - 1:
                print(f"Database connection error (Failed after {max_retries} attempts): {e}")
                raise e
            
            sleep_time = base_delay * (2 ** attempt)
            print(f"Database connection transient error: {e}. Retrying in {sleep_time} seconds (Attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Unexpected database connection error: {e}")
            raise e

# Rate limiting setup
login_attempts = defaultdict(list)
signup_attempts = defaultdict(list)
MAX_ATTEMPTS = 5  # Maximum attempts
WINDOW_SECONDS = 300  # 5 minutes window

def check_rate_limit(ip_address, attempts_dict):
    """Check if IP has exceeded rate limit."""
    now = time.time()
    # Clean old attempts
    attempts_dict[ip_address] = [
        timestamp for timestamp in attempts_dict[ip_address]
        if now - timestamp < WINDOW_SECONDS
    ]
    
    if len(attempts_dict[ip_address]) >= MAX_ATTEMPTS:
        return False, f"Too many attempts. Please try again in {int(WINDOW_SECONDS/60)} minutes."
    
    return True, None

def record_attempt(ip_address, attempts_dict):
    """Record an attempt from IP."""
    attempts_dict[ip_address].append(time.time())

def validate_password(password):
    """
    Validate password strength.
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*...)"
    
    return True, "Password is strong"

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format"
    return True, None

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            has_chat_access BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Coupons table
    c.execute('''
        CREATE TABLE IF NOT EXISTS coupons (
            id SERIAL PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            discount_type TEXT DEFAULT 'free_access',
            discount_value NUMERIC DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration for coupons table: add discount_value if it doesn't exist
    try:
        c.execute("SELECT discount_value FROM coupons LIMIT 1")
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        c = conn.cursor()
        print("Migrating DB: Adding discount_value column to coupons table...")
        c.execute('ALTER TABLE coupons ADD COLUMN discount_value NUMERIC DEFAULT 0')
    except Exception as e:
        conn.rollback()
        c = conn.cursor()
        print(f"Error checking/migrating discount_value column: {e}")
    
    # Conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id),
            session_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            shlokas TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add index for timestamp and user_id to speed up analytics
    c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)')
    
    # Check if session_id column exists (migration for existing DB)
    try:
        c.execute("SELECT session_id FROM conversations LIMIT 1")
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        print("Migrating DB: Adding session_id column...")
        c.execute('ALTER TABLE conversations ADD COLUMN session_id TEXT')
    except Exception as e:
        conn.rollback()
        print(f"Error checking/migrating session_id column: {e}")
    
    # Password reset tokens table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reset_tokens (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id),
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Check if role column exists (migration)
    try:
        c.execute("SELECT role FROM users LIMIT 1")
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        c = conn.cursor()
        print("Migrating DB: Adding role and has_chat_access columns...")
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        c.execute("ALTER TABLE users ADD COLUMN has_chat_access BOOLEAN DEFAULT FALSE")
        c.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except Exception as e:
        conn.rollback()
        c = conn.cursor()
        print(f"Error checking/migrating role column: {e}")

    # Ensure the default admin exists
    admin_email = "abhishek@justlearnindia.in"
    c.execute("SELECT id FROM users WHERE email = %s", (admin_email,))
    if not c.fetchone():
        print(f"Creating default admin: {admin_email}")
        admin_password = generate_password_hash("AdminPassword123!")
        c.execute('''
            INSERT INTO users (name, email, password, role, has_chat_access)
            VALUES (%s, %s, %s, %s, %s)
        ''', ("Admin Abhishek", admin_email, admin_password, "admin", True))
    
    conn.commit()
    conn.close()

def get_user_history(user_id, session_id=None, limit=5):
    """Get recent conversation history for a user, optionally filtered by session."""
    conn = get_db_connection()
    c = conn.cursor()
    
    if session_id:
        # If session_id provided, only get history for THAT session
        c.execute('''
            SELECT question, answer, shlokas, timestamp 
            FROM conversations 
            WHERE user_id = %s AND session_id = %s
            ORDER BY timestamp DESC 
            LIMIT %s
        ''', (user_id, session_id, limit))
    else:
        # Fallback to global history (or maybe just empty if we want strict sessions?)
        c.execute('''
            SELECT question, answer, shlokas, timestamp 
            FROM conversations 
            WHERE user_id = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        ''', (user_id, limit))
        
    history = c.fetchall()
    conn.close()
    
    # Format history for LLM context
    formatted_history = []
    for q, a, shlokas, ts in reversed(history):  # Reverse to get chronological order
        formatted_history.append({
            'question': q,
            'answer': a,
            'timestamp': ts
        })
    return formatted_history

def save_conversation(user_id, question, answer, shlokas, session_id=None):
    """Save a conversation to the database."""
    conn = get_db_connection()
    c = conn.cursor()
    shlokas_json = json.dumps(shlokas) if shlokas else None
    c.execute('''
        INSERT INTO conversations (user_id, session_id, question, answer, shlokas)
        VALUES (%s, %s, %s, %s, %s)
    ''', (user_id, session_id, question, answer, shlokas_json))
    conn.commit()
    conn.close()

def generate_reset_token():
    """Generate a secure random token."""
    import secrets
    return secrets.token_urlsafe(32)

def create_reset_token(user_id):
    """Create a password reset token for a user."""
    token = generate_reset_token()
    from datetime import datetime, timedelta
    
    # Token expires in 1 hour
    expires_at = datetime.now() + timedelta(hours=1)
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO reset_tokens (user_id, token, expires_at)
        VALUES (%s, %s, %s)
    ''', (user_id, token, expires_at.isoformat()))
    conn.commit()
    conn.close()
    
    return token

def validate_reset_token(token):
    """Validate a reset token and return user_id if valid."""
    from datetime import datetime
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT user_id, expires_at, used 
        FROM reset_tokens 
        WHERE token = %s
    ''', (token,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return None, "Invalid reset token"
    
    user_id, expires_at, used = result
    
    if used:
        return None, "This reset link has already been used"
    
    # Check if token has expired
    if isinstance(expires_at, str):
        expires_datetime = datetime.fromisoformat(expires_at)
    else:
        expires_datetime = expires_at
    if datetime.now() > expires_datetime:
        return None, "This reset link has expired"
    
    return user_id, None

def mark_token_used(token):
    """Mark a reset token as used."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE reset_tokens 
        SET used = TRUE 
        WHERE token = %s
    ''', (token,))
    conn.commit()
    conn.close()

@app.route('/api/history', methods=['GET'])
def get_history_api():
    """Fetch all history for a specific user to display in the UI"""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID is required', 'success': False}), 400
            
        # Get all history up to 50 conversations for the sidebar
        raw_history = get_user_history(user_id, limit=50)
        
        return jsonify({
            'success': True,
            'history': raw_history
        })
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'error': 'Failed to fetch history', 'success': False}), 500

# Initialize DB
init_db()

@app.route('/api/signup', methods=['POST'])
def signup():
    # Get client IP for rate limiting
    client_ip = request.remote_addr
    
    # Check rate limit
    allowed, error_msg = check_rate_limit(client_ip, signup_attempts)
    if not allowed:
        return jsonify({'error': error_msg, 'success': False}), 429
    
    # Record this attempt
    record_attempt(client_ip, signup_attempts)
    
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    # Validate required fields
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required', 'success': False}), 400
    
    # Validate name length
    if len(name) < 2:
        return jsonify({'error': 'Name must be at least 2 characters', 'success': False}), 400
    
    if len(name) > 100:
        return jsonify({'error': 'Name is too long', 'success': False}), 400

    # Validate email format
    email_valid, email_error = validate_email(email)
    if not email_valid:
        return jsonify({'error': email_error, 'success': False}), 400

    # Validate password strength
    password_valid, password_error = validate_password(password)
    if not password_valid:
        return jsonify({'error': password_error, 'success': False}), 400

    # Hash password
    hashed_pw = generate_password_hash(password)

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, hashed_pw))
        conn.commit()
        conn.close()
        
        print(f"New user registered: {email}")
        return jsonify({'message': 'Account created successfully!', 'success': True}), 201
    except psycopg2.IntegrityError:
        return jsonify({'error': 'This email is already registered', 'success': False}), 409
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'error': 'Registration failed. Please try again.', 'success': False}), 500

@app.route('/api/login', methods=['POST'])
def login():
    # Get client IP for rate limiting
    client_ip = request.remote_addr
    
    # Check rate limit
    allowed, error_msg = check_rate_limit(client_ip, login_attempts)
    if not allowed:
        return jsonify({'error': error_msg, 'success': False}), 429
    
    # Record this attempt
    record_attempt(client_ip, login_attempts)
    
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required', 'success': False}), 400
    
    # Validate email format
    email_valid, email_error = validate_email(email)
    if not email_valid:
        return jsonify({'error': 'Invalid email format', 'success': False}), 400

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, name, email, password, role, has_chat_access FROM users WHERE email = %s', (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            print(f"Successful login: {email}")
            return jsonify({
                'message': 'Login successful',
                'success': True,
                'user': {
                    'id': user[0],
                    'name': user[1],
                    'email': user[2],
                    'role': user[4],
                    'has_chat_access': user[5]
                }
            }), 200
        else:
            print(f"Failed login attempt: {email}")
            return jsonify({'error': 'Invalid email or password', 'success': False}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed. Please try again.', 'success': False}), 500

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    """Request a password reset token."""
    data = request.get_json()
    email = data.get('email', '').strip()
    
    if not email:
        return jsonify({'error': 'Email is required', 'success': False}), 400
    
    # Validate email format
    email_valid, email_error = validate_email(email)
    if not email_valid:
        return jsonify({'error': 'Invalid email format', 'success': False}), 400
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = %s', (email,))
        user = c.fetchone()
        conn.close()
        
        # Always return success to prevent email enumeration
        # But only create token if user exists
        if user:
            user_id = user[0]
            token = create_reset_token(user_id)
            
            # In production, send this token via email
            # For now, we'll return it in the response (NOT SECURE FOR PRODUCTION)
            print(f"Password reset requested for: {email}")
            print(f"Reset token: {token}")
            
            # TODO: Send email with reset link
            # reset_link = f"http://localhost:3000/reset-password?token={token}"
            
            return jsonify({
                'success': True,
                'message': 'If an account exists with this email, a reset link has been sent.'
            }), 200
        else:
            # Return same message to prevent email enumeration
            return jsonify({
                'success': True,
                'message': 'If an account exists with this email, a reset link has been sent.'
            }), 200
            
    except Exception as e:
        print(f"Forgot password error: {e}")
        return jsonify({'error': 'Request failed. Please try again.', 'success': False}), 500

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """Reset password using a valid token."""
    data = request.get_json()
    token = data.get('token', '').strip()
    new_password = data.get('password', '')
    
    if not token or not new_password:
        return jsonify({'error': 'Token and new password are required', 'success': False}), 400
    
    # Validate password strength
    password_valid, password_error = validate_password(new_password)
    if not password_valid:
        return jsonify({'error': password_error, 'success': False}), 400
    
    # Validate token
    user_id, error = validate_reset_token(token)
    if error:
        return jsonify({'error': error, 'success': False}), 400
    
    try:
        # Update password
        hashed_pw = generate_password_hash(new_password)
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('UPDATE users SET password = %s WHERE id = %s', (hashed_pw, user_id))
        conn.commit()
        conn.close()
        
        # Mark token as used
        mark_token_used(token)
        
        print(f"Password reset successful for user ID: {user_id}")
        return jsonify({
            'success': True,
            'message': 'Password has been reset successfully. You can now log in with your new password.'
        }), 200
        
    except Exception as e:
        print(f"Reset password error: {e}")
        return jsonify({'error': 'Password reset failed. Please try again.', 'success': False}), 500

@app.route('/api/grant-access', methods=['POST'])
def grant_chat_access_after_payment():
    """
    Grant chat access to a user after successful payment/checkout.
    Called automatically from frontend after purchase is complete.
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'success': False}), 400

        conn = get_db_connection()
        c = conn.cursor()

        # Verify user exists
        c.execute('SELECT id, name, email FROM users WHERE id = %s', (user_id,))
        user = c.fetchone()

        if not user:
            conn.close()
            return jsonify({'error': 'User not found', 'success': False}), 404

        # Grant chat access
        c.execute('UPDATE users SET has_chat_access = TRUE WHERE id = %s', (user_id,))
        conn.commit()
        conn.close()

        print(f"✅ Chat access granted to user: {user[2]} (ID: {user_id})")
        return jsonify({
            'success': True,
            'message': 'Chat access granted successfully',
            'has_chat_access': True
        }), 200

    except Exception as e:
        print(f"Grant access error: {e}")
        return jsonify({'error': 'Failed to grant access. Please try again.', 'success': False}), 500


# --- Admin Endpoints ---

def admin_required(f):
    """Decorator to require admin role."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = request.args.get('admin_id') or request.get_json().get('admin_id')
        if not user_id:
            return jsonify({'error': 'Admin ID is required', 'success': False}), 401
        
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT role FROM users WHERE id = %s', (user_id,))
            user = c.fetchone()
            conn.close()
            
            if not user or user[0] != 'admin':
                return jsonify({'error': 'Admin privilege required', 'success': False}), 403
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    try:
        conn = get_db_connection()
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT id, name, email, role, has_chat_access, created_at FROM users ORDER BY created_at DESC')
        users = c.fetchall()
        conn.close()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/create-admin', methods=['POST'])
@admin_required
def create_admin():
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    name = data.get('name', 'Admin').strip()
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required', 'success': False}), 400
        
    hashed_pw = generate_password_hash(password)
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO users (name, email, password, role, has_chat_access)
            VALUES (%s, %s, %s, 'admin', True)
        ''', (name, email, hashed_pw))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Admin created successfully', 'success': True})
    except psycopg2.IntegrityError:
        return jsonify({'error': 'Email already exists', 'success': False}), 409
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/grant-access', methods=['POST'])
@admin_required
def grant_access():
    data = request.get_json()
    user_email = data.get('email', '').strip()
    has_access = data.get('has_access', True)
    temporary_password = data.get('temporary_password') # Optional if user already exists
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check if user exists
        c.execute('SELECT id FROM users WHERE email = %s', (user_email,))
        user = c.fetchone()
        
        if not user:
            if not temporary_password:
                return jsonify({'error': 'User does not exist and no temporary password provided', 'success': False}), 400
            
            # Create user if doesn't exist
            hashed_pw = generate_password_hash(temporary_password)
            c.execute('''
                INSERT INTO users (name, email, password, has_chat_access)
                VALUES (%s, %s, %s, %s)
            ''', (user_email.split('@')[0], user_email, hashed_pw, has_access))
            message = "User created and access granted"
        else:
            # Update existing user
            c.execute('UPDATE users SET has_chat_access = %s WHERE email = %s', (has_access, user_email))
            message = f"Access {'granted' if has_access else 'revoked'} successfully"
            
        conn.commit()
        conn.close()
        return jsonify({'message': message, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@admin_required
def get_analytics():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Total users
        c.execute('SELECT COUNT(*) FROM users')
        total_users = c.fetchone()[0]
        
        # Users used today
        c.execute('''
            SELECT COUNT(DISTINCT user_id) 
            FROM conversations 
            WHERE timestamp >= CURRENT_DATE
        ''')
        today_users = c.fetchone()[0]
        
        # Total conversations
        c.execute('SELECT COUNT(*) FROM conversations')
        total_convs = c.fetchone()[0]
        
        # Convs today
        c.execute('SELECT COUNT(*) FROM conversations WHERE timestamp >= CURRENT_DATE')
        today_convs = c.fetchone()[0]
        
        conn.close()
        return jsonify({
            'success': True,
            'analytics': {
                'total_users': total_users,
                'today_users': today_users,
                'total_conversations': total_convs,
                'today_conversations': today_convs
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/conversations', methods=['GET'])
@admin_required
def get_all_conversations():
    try:
        limit = request.args.get('limit', 50)
        offset = request.args.get('offset', 0)
        
        conn = get_db_connection()
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('''
            SELECT c.id, c.user_id, u.name as user_name, u.email as user_email, 
                   c.question, c.answer, c.timestamp 
            FROM conversations c 
            JOIN users u ON c.user_id = u.id 
            ORDER BY c.timestamp DESC 
            LIMIT %s OFFSET %s
        ''', (limit, offset))
        conversations = c.fetchall()
        conn.close()
        
        return jsonify({'success': True, 'conversations': conversations})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/conversation-users', methods=['GET'])
@admin_required
def get_conversation_users():
    try:
        conn = get_db_connection()
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('''
            SELECT u.id, u.name, u.email, 
                   COUNT(c.id) as conversation_count, 
                   MAX(c.timestamp) as last_active
            FROM users u
            JOIN conversations c ON u.id = c.user_id
            GROUP BY u.id, u.name, u.email
            ORDER BY last_active DESC
        ''')
        users = c.fetchall()
        conn.close()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/user-conversations/<int:user_id>', methods=['GET'])
@admin_required
def get_specific_user_conversations(user_id):
    try:
        conn = get_db_connection()
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('''
            SELECT id, question, answer, timestamp, session_id
            FROM conversations 
            WHERE user_id = %s
            ORDER BY timestamp DESC
        ''', (user_id,))
        conversations = c.fetchall()
        
        # Get user info too
        c.execute('SELECT name, email FROM users WHERE id = %s', (user_id,))
        user_info = c.fetchone()
        
        conn.close()
        return jsonify({
            'success': True, 
            'conversations': conversations,
            'user': user_info
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/coupons', methods=['GET'])
@admin_required
def get_coupons():
    try:
        conn = get_db_connection()
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT id, code, discount_type, discount_value, is_active, created_at FROM coupons ORDER BY created_at DESC')
        coupons = c.fetchall()
        conn.close()
        return jsonify({'success': True, 'coupons': coupons})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/coupons', methods=['POST'])
@admin_required
def add_coupon():
    data = request.get_json()
    code = data.get('code', '').strip().upper()
    discount_type = data.get('discount_type', 'free_access')
    discount_value = data.get('discount_value', 0)
    
    if not code:
        return jsonify({'error': 'Coupon code is required', 'success': False}), 400
        
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO coupons (code, discount_type, discount_value, is_active)
            VALUES (%s, %s, %s, TRUE)
        ''', (code, discount_type, discount_value))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Coupon added successfully', 'success': True})
    except psycopg2.IntegrityError:
        return jsonify({'error': 'Coupon code already exists', 'success': False}), 409
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/validate-coupon', methods=['POST'])
def validate_coupon():
    try:
        data = request.get_json()
        code = data.get('code', '').strip().upper()
        
        if not code:
            return jsonify({'error': 'Coupon code is required', 'success': False}), 400
            
        conn = get_db_connection()
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT code, discount_type, discount_value, is_active FROM coupons WHERE code = %s', (code,))
        coupon = c.fetchone()
        conn.close()
        
        if not coupon:
            return jsonify({'error': 'Invalid coupon code', 'success': False}), 404
            
        if not coupon['is_active']:
            return jsonify({'error': 'Coupon is no longer active', 'success': False}), 400
            
        return jsonify({
            'success': True, 
            'coupon': {
                'code': coupon['code'],
                'discount_type': coupon['discount_type'],
                'discount_value': float(coupon['discount_value']) if coupon['discount_value'] else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/coupons/<int:coupon_id>/toggle', methods=['POST'])
@admin_required
def toggle_coupon_status(coupon_id):
    try:
        data = request.get_json()
        is_active = data.get('is_active')
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('UPDATE coupons SET is_active = %s WHERE id = %s', (is_active, coupon_id))
        conn.commit()
        conn.close()
        return jsonify({'message': f'Coupon {"activated" if is_active else "deactivated"} successfully', 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/admin/coupons/<int:coupon_id>', methods=['DELETE'])
@admin_required
def delete_coupon(coupon_id):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('DELETE FROM coupons WHERE id = %s', (coupon_id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Coupon deleted successfully', 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Talk to Krishna - Web API Server")
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("Open website/index.html in your browser to use the web interface\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Set to True for development
    )
