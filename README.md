# 🏥 Doctor's Co-Pilot — Production-Grade Clinical RAG Engine

Doctor's Co-Pilot is a Retrieval-Augmented Generation (RAG) system designed for clinical environments. It allows medical practitioners to query complex patient records, discharge summaries, and clinical notes in natural language via an interactive dashboard. 

This engine is architected to address real-world deployment challenges including **strict patient data isolation (HIPAA alignment)**, **dynamic multi-format document ingestion**, and **stateful multi-turn conversation memory**.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────┐
                       │   Doctor's Web Dashboard UI   │
                       └──────────────┬───────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼ (User Query + Patient Filter + Session ID)   ▼ (Upload PDF/TXT Clinical Note)
     ┌───────────────────┐                          ┌───────────────────┐
     │ 1. Query Condense │                          │ 1. Text Extractor │
     │  (Gemini Flask)   │                          │  (pypdf Parser)   │
     └─────────┬─────────┘                          └─────────┬─────────┘
               │ (Rephrases query using history)              │ (Extracts text content)
               ▼                                              ▼
     ┌───────────────────┐                          ┌───────────────────┐
     │ 2. Embed Query    │                          │ 2. Text Chunker   │
     │ (gemini-embed-001)│                          │ (150w/30w overlap)│
     └─────────┬─────────┘                          └─────────┬─────────┘
               │ (Generates 3072-dim vector)                  │ (Granular word chunks)
               ▼                                              ▼
     ┌───────────────────┐                          ┌───────────────────┐
     │ 3. Metadata query │                          │ 3. Embed & Upsert │
     │ (Pinecone Vector) │                          │ (Pinecone Index)  │
     └─────────┬─────────┘                          └───────────────────┘
               │ (Filters by Active Patient ID)
               ▼ (Retrieves top-K source chunks)
     ┌───────────────────┐
     │ 4. Grounded Gen   │
     │(gemini-2.5-flash) │
     └─────────┬─────────┘
               │ (Grounded answer strictly via sources)
               ▼
     ┌───────────────────┐
     │ 5. Response Render│
     │  (Chat Bubbles)   │
     └───────────────────┘
```

---

## ⚡ Key Production-Grade Features

### 1. 🔒 HIPAA-Aligned Patient Data Isolation (Metadata Filtering)
In a hospital setting, search queries must never leak data between patients. The backend enforces security by binding Pinecone vector queries to metadata filters: `filter={"patient_id": {"$eq": patient_id}}`. This guarantees that vector retrieval is partitioned exclusively to the active patient, even if other patient records have similar semantic symptoms.

### 2. 📄 Dynamic Document Ingestion Pipeline
Allows doctors to upload clinical summaries in PDF or Plain Text formats directly from the dashboard. The backend dynamically:
* Extracts raw text from multi-page PDFs using `pypdf`.
* Splits text using a **sliding-window word chunker** (150 words per chunk with 30-word overlaps) to preserve local semantic context.
* Generates 3072-dimensional embeddings via Gemini and batch-upserts them to Pinecone with structured metadata.

### 3. 🧠 Context-Aware Query Condensing (Conversational Memory)
Handles multi-turn dialogue naturally. If a doctor asks a follow-up question (e.g., *"What is his dosage?"*), a pre-retrieval loop rephrases it into a standalone query (e.g., *"What is patient PT-8829's dosage for Amiodarone?"*) using the conversation's history. This rephrased query is what searches the vector database, resolving retrieval context limits.

### 4. 🔬 Explainable AI Tracing
The chat UI renders:
* **The Condensed Query:** Showing how the conversation history was compiled before searching.
* **Vector Source Citations:** Detailed cards for each matching chunk, including dates, departments, prescribing doctors, and cosine similarity scores.

---

## 🛠️ Tech Stack

| Layer | Technology | Description |
|---|---|---|
| **Vector DB** | Pinecone (Serverless) | Fast metadata-filtered similarity search |
| **Embeddings** | Gemini `embedding-001` | High-quality 3072-dimension semantic vectors |
| **LLM Generation**| Gemini `gemini-2.5-flash` | Ultra-fast grounded generation with rate-limiting retries |
| **Backend** | Flask (Python 3.10+) | Secure file uploads, session memory, and RAG execution |
| **Frontend** | Vanilla HTML / CSS / JS | Cyberpunk glassmorphic medical dashboard UI |

---

## 🚀 Installation & Setup

### 1. Clone & Configure Environment
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your-pinecone-api-key
GEMINI_API_KEY=your-gemini-api-key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Vector Database Initialization (Once)
Populate the vector index with sample patient records:
```bash
python setup_pinecone.py
```

### 4. Start the Application Server
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🧪 Testing the Advanced Capabilities

1. **Test Conversational Memory:**
   * Enter `PT-8829` in the **Active Patient Filter**.
   * Ask: *"Why was this patient admitted in January 2024?"*
   * Follow up with: *"What procedure was performed for it?"*
   * *Notice:* The UI displays the condensed search query indicating it merged the context of the previous turn.

2. **Test Isolation (HIPAA Compliance):**
   * Clear the chat session.
   * Set the filter to `PT-1234` (Diabetes history).
   * Ask: *"Summarize his cardiac complications."*
   * *Result:* The engine will correctly report no records found, because `PT-8829`'s cardiac notes are locked out by the filter.

3. **Test PDF Ingestion:**
   * Fill out the **Ingest Patient File** form (e.g., ID: `PT-9999`, Doctor: `Dr. Sen`, Dept: `Oncology`).
   * Upload a medical PDF or text file.
   * Click **Upload & Index Chunks**.
   * Query the system about the contents of the uploaded note under the active filter `PT-9999`.
