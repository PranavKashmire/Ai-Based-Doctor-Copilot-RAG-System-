"""
app.py
------
Upgraded Flask web server for the Doctor's Co-Pilot RAG system.
Serves a beautiful, interactive chat interface, handles dynamic patient document ingestion
(PDF/TXT), enforces strict patient data isolation (metadata filtering), and supports
stateful conversational session memory.

Usage:
    set PINECONE_API_KEY=your-pinecone-api-key
    set GEMINI_API_KEY=your-gemini-api-key
    python app.py
"""

import os
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pinecone import Pinecone
from google import genai
import pypdf
from flashrank import Ranker, RerankRequest

# Load API keys from .env file
load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
INDEX_NAME = "doctor-copilot"
EMBEDDING_MODEL = "gemini-embedding-001"  # 3072 dimensions
GENERATION_MODEL = "gemini-2.5-flash"

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Please set PINECONE_API_KEY and GEMINI_API_KEY environment variables.")

# Initialize clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize FlashRank reranker (loads default lightweight ONNX cross-encoder)
print("[INFO] Initializing FlashRank Reranking engine...")
flash_ranker = Ranker()
print("[SUCCESS] FlashRank ready!")

app = Flask(__name__)

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Conversational session memory store
# Format: {session_id: [{"role": "user"/"model", "content": str}]}
chat_sessions = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────────────────────────
# Core RAG Helpers
# ─────────────────────────────────────────────
def condense_query(user_query: str, session_id: str) -> str:
    """
    If there is chat history, use Gemini to rephrase the query into a standalone, 
    context-rich search query. Otherwise, return the query as-is.
    """
    if not session_id or session_id not in chat_sessions or not chat_sessions[session_id]:
        return user_query
        
    history = chat_sessions[session_id]
    
    # Format history for prompt (limit to last 5 turns to prevent token bloat)
    history_lines = []
    for turn in history[-5:]:
        role_label = "Doctor" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{role_label}: {turn['content']}")
    history_str = "\n".join(history_lines)
    
    condense_prompt = f"""Given the following conversation history between a Doctor and an AI Assistant, and a new follow-up question from the Doctor, rephrase the follow-up question into a standalone query that can be used for semantic search in patient clinical notes.
The rephrased query must explicitly include patient references, medical terms, and specific subjects mentioned in the history so it can be searched independently.

CONVERSATION HISTORY:
{history_str}

FOLLOW-UP QUESTION: {user_query}

Provide ONLY the rephrased standalone query. Do not include any preambles, explanations, or conversational text."""

    try:
        response = gemini_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=condense_prompt,
        )
        condensed = response.text.strip()
        print(f"[INFO] Query Condensed: '{user_query}' -> '{condensed}'")
        return condensed
    except Exception as e:
        print(f"[WARN] Error condensing query: {e}. Using original query.")
        return user_query


def doctor_copilot_query(user_query: str, patient_id: str = None, session_id: str = None, top_k: int = 5) -> dict:
    """
    Execute the Doctor's Co-Pilot RAG workflow:
    1. Rephrases query using chat history if available (Query Condensing).
    2. Embeds the query and performs a similarity search on Pinecone (retrieves top-25 candidates).
    3. Reranks the candidates using FlashRank cross-encoder model down to the top-5.
    4. Prompts Gemini with reranked notes and history to generate a medical summary.
    5. Saves the turn to conversational history.
    """
    print(f"\n[INFO] Original Query: {user_query}")
    if patient_id:
        print(f"[INFO] Filtering for Patient: {patient_id}")

    # ── STEP 1: Query Condensing (Context Rewriting) ──
    search_query = condense_query(user_query, session_id)
    
    # ── STEP 2: Embed and Retrieve Candidates ──
    query_embedding_result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=search_query,
    )
    query_embedding = query_embedding_result.embeddings[0].values

    # Setup Pinecone query args - Fetch 25 candidates for reranking
    query_kwargs = {
        "vector": query_embedding,
        "top_k": 25,
        "include_metadata": True
    }
    
    # Enforce medical isolation (data security) via Pinecone metadata filtering
    if patient_id:
        query_kwargs["filter"] = {"patient_id": {"$eq": patient_id}}

    search_results = index.query(**query_kwargs)

    # ── STEP 3: FlashRank Reranking (Cross-Encoder Optimization) ──
    passages = []
    for match in search_results.matches:
        passages.append({
            "id": match.id,
            "text": match.metadata.get("text", ""),
            "meta": {
                "patient_id": match.metadata.get("patient_id", "Unknown"),
                "doctor_name": match.metadata.get("doctor_name", "Unknown"),
                "date": match.metadata.get("date", "Unknown"),
                "department": match.metadata.get("department", "Unknown"),
            }
        })

    if not passages:
        # Save user query to history even on empty match to maintain thread state
        if session_id:
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            chat_sessions[session_id].append({"role": "user", "content": user_query})
            chat_sessions[session_id].append({"role": "model", "content": "No relevant patient records found for this query."})
            
        return {
            "summary": "No relevant patient records found for this query.",
            "chunks": [],
            "condensed_query": search_query
        }

    # Execute FlashRank Reranker on candidates
    print(f"[INFO] Reranking {len(passages)} chunks using FlashRank cross-encoder...")
    rerank_request = RerankRequest(query=search_query, passages=passages)
    reranked_results = flash_ranker.rerank(rerank_request)

    # Slice down to top_k (default top-5) to optimize context window & tokens
    top_reranked = reranked_results[:top_k]
    print(f"[SUCCESS] Selected top-{len(top_reranked)} reranked chunks.")

    # Extract reranked text & compile metadata
    retrieved_chunks = []
    chunks_metadata = []
    for rank_idx, match in enumerate(top_reranked):
        chunk_text = match.get("text", "")
        retrieved_chunks.append(chunk_text)
        
        meta = match.get("meta", {})
        chunks_metadata.append({
            "patient_id": meta.get("patient_id", "Unknown"),
            "doctor_name": meta.get("doctor_name", "Unknown"),
            "date": meta.get("date", "Unknown"),
            "department": meta.get("department", "Unknown"),
            "relevance": round(float(match.get("score", 0)), 4),
            "text": chunk_text,
            "rank": rank_idx + 1
        })

    # ── STEP 4: Context Compilation & Generation ──
    context = "\n\n---\n\n".join(
        [f"[Clinical Note {i+1}]: {chunk}"
         for i, chunk in enumerate(retrieved_chunks)]
    )

    # Fetch conversational history context
    history_context = ""
    if session_id and session_id in chat_sessions and chat_sessions[session_id]:
        history_lines = []
        for turn in chat_sessions[session_id][-5:]:
            role_label = "Doctor" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {turn['content']}")
        history_context = "\nRecent Conversation History:\n" + "\n".join(history_lines) + "\n"

    prompt = f"""You are a medical AI assistant at Rungta Hospital. A doctor is asking 
a question about a patient's medical history. Use ONLY the clinical notes provided below 
and the recent conversation history to answer. Do not hallucinate or make up any medical information.

If the provided notes do not contain enough information to fully answer the question, 
clearly state what information is available and what is missing.

--- RETRIEVED CLINICAL NOTES ---
{context}
--- END OF CLINICAL NOTES ---
{history_context}
DOCTOR'S QUESTION: {user_query}

Please provide a clear, concise, and medically accurate summary based strictly on the 
clinical notes and context above. Do not refer to patient records not provided."""

    # Call Gemini with exponential backoff for rate limits
    max_retries = 3
    summary = ""
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
            )
            summary = response.text
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"[WARN] Gemini Rate limit hit. Retrying in 45s... ({attempt+1}/{max_retries})")
                time.sleep(45)
            else:
                raise e

    # ── STEP 5: Store history (max 10 entries / 5 turns) ──
    if session_id:
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        chat_sessions[session_id].append({"role": "user", "content": user_query})
        chat_sessions[session_id].append({"role": "model", "content": summary})
        chat_sessions[session_id] = chat_sessions[session_id][-10:]

    print(f"[SUCCESS] Summary generated!")
    return {
        "summary": summary,
        "chunks": chunks_metadata,
        "condensed_query": search_query
    }


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    """Serve the updated chat interface."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """API endpoint for RAG queries."""
    data = request.get_json()
    user_query = data.get("query", "").strip()
    patient_id = data.get("patient_id", "").strip() or None
    session_id = data.get("session_id", "").strip() or None

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        result = doctor_copilot_query(user_query, patient_id=patient_id, session_id=session_id)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """API endpoint to parse, chunk, embed, and index patient PDFs or TXT files."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    patient_id = request.form.get("patient_id", "").strip()
    doctor_name = request.form.get("doctor_name", "").strip() or "Unknown"
    department = request.form.get("department", "").strip() or "General Medicine"
    date = request.form.get("date", "").strip() or time.strftime("%Y-%m-%d")
    
    if not patient_id:
        return jsonify({"error": "Patient ID is required for data isolation"}), 400
        
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Unsupported file type. Only PDF and TXT files are accepted"}), 400
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ── Text Extraction ──
        text = ""
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext == 'pdf':
            reader = pypdf.PdfReader(filepath)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
            
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)
            
        if not text.strip():
            return jsonify({"error": "Failed to extract readable text from document"}), 400
            
        # ── Semantic Text Chunking with Overlap (Word-based) ──
        words = text.split()
        chunk_size = 150  # Words per chunk
        chunk_overlap = 30  # Overlap words
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += (chunk_size - chunk_overlap)
            
        if not chunks:
            return jsonify({"error": "Document length is too short to generate chunks"}), 400

        print(f"[INFO] Processing document '{filename}': generated {len(chunks)} chunks for Patient '{patient_id}'")
        
        # ── Embed chunks in batches ──
        batch_size = 10
        all_embeddings = []
        for start_idx in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start_idx:start_idx + batch_size]
            embed_result = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch_chunks,
            )
            all_embeddings.extend([e.values for e in embed_result.embeddings])
            
        # ── Upsert Vectors to Pinecone ──
        vectors = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, all_embeddings)):
            chunk_id = f"upload-{patient_id}-{idx}-{int(time.time())}"
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "patient_id": patient_id,
                    "doctor_name": doctor_name,
                    "department": department,
                    "date": date,
                    "text": chunk_text
                }
            })
            
        # Batch upload to Pinecone (limit 50 per batch)
        for offset in range(0, len(vectors), 50):
            index.upsert(vectors=vectors[offset:offset + 50])
            
        print(f"[SUCCESS] Ingested {len(chunks)} vectors to Pinecone for Patient: {patient_id}")
        return jsonify({
            "message": f"Successfully ingested {len(chunks)} document chunks to Pinecone for Patient {patient_id}.",
            "chunks_count": len(chunks)
        })
        
    except Exception as e:
        print(f"[ERROR] Error during document upload: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Endpoint to clear conversational history for the current session."""
    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    if session_id and session_id in chat_sessions:
        del chat_sessions[session_id]
        print(f"[INFO] Reset conversational history for Session ID: {session_id}")
        return jsonify({"success": True, "message": "History cleared"})
    return jsonify({"success": False, "error": "Invalid session ID"}), 400


if __name__ == "__main__":
    print("[INFO] Doctor's Co-Pilot is running at http://localhost:5000")
    app.run(debug=True, port=5000)
