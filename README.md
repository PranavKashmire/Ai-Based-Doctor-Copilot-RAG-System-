# 🏥 Project: Doctor's Co-Pilot

## 🏷️ Project Name
**Doctor's Co-Pilot — Production-Grade Clinical RAG (Retrieval-Augmented Generation) Engine**

---

## 🎯 Overview
**Doctor's Co-Pilot** is an advanced, domain-specific Retrieval-Augmented Generation (RAG) system engineered for clinical healthcare environments. It provides medical practitioners with an interactive, natural language chat interface to query patient clinical notes, discharge summaries, and lab results. 

The system leverages state-of-the-art Large Language Models (LLMs), serverless vector databases, and cross-encoder reranking models to retrieve, rank, and summarize patient medical records accurately.

---

## ⚠️ The Problems It Solves
Deploying AI in healthcare environments introduces strict engineering constraints that simple RAG systems cannot handle. Doctor's Co-Pilot directly solves these production challenges:

1. **🔒 Cross-Patient Data Leakage (HIPAA Isolation):**
   * *Problem:* In standard semantic vector search, a query about a symptom (e.g., *"What is his heart rate?"*) might retrieve notes belonging to other patients who have similar symptoms. This violates HIPAA data privacy rules.
   * *Solution:* Doctor's Co-Pilot implements **strict metadata filtering**. Vector database queries are isolated via user-scoped metadata constraints: `filter={"patient_id": {"$eq": patient_id}}`, ensuring data boundaries are absolute.
2. **🧠 Conversational Context Loss in Multi-Turn RAG:**
   * *Problem:* Follow-up questions (e.g. *"What is his dosage?"*) lack keywords like the patient ID or medication name, leading to poor vector database retrieval.
   * *Solution:* Employs **pre-retrieval query condensing** using LLM memory to automatically rewrite conversational queries into standalone context-rich search terms before searching the index.
3. **💥 LLM Distraction & Token Bloat:**
   * *Problem:* Passing too many context chunks to the LLM increases token costs and degrades generation quality (known as the "lost in the middle" effect).
   * *Solution:* Introduces a **two-stage retrieval pipeline**. The system queries 25 candidates from Pinecone and rerank-filters them down to the top 5 using a high-precision local cross-encoder model (**FlashRank**), reducing context sizes by over 30%.
4. **🚫 Medical Hallucinations:**
   * *Problem:* Standard LLMs generate answers based on pre-training data, which is unacceptable for medical diagnostics.
   * *Solution:* Employs a strict medical grounding system prompt that forces the generator to answer **ONLY** using the retrieved notes, explicitly list source citations, and refuse to answer if information is missing.

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
               ▼ (Retrieves top-25 candidate chunks)
     ┌───────────────────┐
     │ 4. Cross-Rerank   │
     │ (FlashRank Engine)│
     └─────────┬─────────┘
               │ (Reranks down to top-5 most relevant)
               ▼
     ┌───────────────────┐
     │ 5. Grounded Gen   │
     │(gemini-2.5-flash) │
     └─────────┬─────────┘
               │ (Grounded answer strictly via sources)
               ▼
     ┌───────────────────┐
     │ 6. Response Render│
     │ (Explainable UI)  │
     └───────────────────┘
```

---

## 🛠️ Tech Stack
* **LLM (Generation):** Google Gemini SDK (`gemini-2.5-flash`)
* **Vector Embeddings:** Google Gemini SDK (`gemini-embedding-001`, 3072 dimensions)
* **Vector Database:** Pinecone (Serverless cosine index)
* **Reranking Engine:** FlashRank (Lightweight ONNX Cross-Encoder: `ms-marco-TinyBERT-L-2-v2`)
* **Backend Framework:** Python Flask (with `werkzeug` and `pypdf`)
* **Frontend UI:** Glassmorphic Dark-Themed Dashboard (HTML, CSS, Vanilla JS)

---

## 📖 Detailed Explanation of Operations

### 1. The Ingestion Pipeline (Dynamic Document Parsing)
* **File Upload:** The Flask endpoint `/api/upload` accepts a PDF or TXT file along with patient metadata (Patient ID, Doctor Name, Department, Date).
* **Text Extraction:** Raw text is extracted from multi-page files page-by-page.
* **Sliding-Window Chunking:** Chunks are split using a character/word boundary splitter set to 150 words per chunk with a 30-word overlap. This overlap ensures clinical notes do not lose critical diagnostic sentences at the boundaries of the split text.
* **Batch Embedding:** Chunks are batch-sent to the embedding API and mapped to 3072-dimension vectors.
* **Metadata Association:** The vectors are upserted into Pinecone with attached metadata keys:
  ```json
  {
    "id": "upload-PT-9999-0-1783411",
    "values": [0.012, -0.045, ...],
    "metadata": {
      "patient_id": "PT-9999",
      "doctor_name": "Dr. Sen",
      "department": "Oncology",
      "date": "2024-05-05",
      "text": "Patient PT-9999 was started on Cisplatin..."
    }
  }
  ```

### 2. The Query Pipeline (Explainable Answering)
* **Active Filter Scoping:** The UI captures the selected active patient filter input and scopes the search parameters.
* **Conversational Context Condensing:** If a follow-up query is detected within the active Session ID, the backend rephrases the follow-up question (e.g. *"What is his dosage?"*) to merge historical reference data into a standalone search query.
* **Vector Vectorization:** The condensed query is translated into a 3072-dimensional search vector.
* **Candidate Retrieval (Pinecone):** Pinecone performs vector retrieval but limits the candidate space by matching `patient_id` metadata. It retrieves the top 25 candidate chunks.
* **Cross-Encoder Reranking (FlashRank):** The 25 candidates are cross-scored against the condensed query on the local server CPU. Chunks are re-ordered and sliced down to the **top 5**.
* **Prompt Assembly:** The 5 context chunks, history logs, and prompt guidelines are compiled into a custom prompt instructing the LLM to act as a grounded assistant.
* **UI citations Rendering:** The chat renders the response, the condensed query, and CITATION tags displaying rank and scoring indicators.

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
```bash
python setup_pinecone.py
```

### 4. Start the Application Server
```bash
python app.py
```
Open **http://localhost:5000** in your browser.
