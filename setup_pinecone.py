"""
setup_pinecone.py
-----------------
This script creates a Pinecone index and populates it with
FAKE patient discharge summary data for demonstration purposes.

Run this ONCE before using doctor_copilot.py.

Usage:
    set PINECONE_API_KEY=your-pinecone-api-key
    set GEMINI_API_KEY=your-gemini-api-key
    python setup_pinecone.py
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from google import genai

# Load API keys from .env file
load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
INDEX_NAME = "doctor-copilot"
EMBEDDING_MODEL = "gemini-embedding-001"  # 3072 dimensions
EMBEDDING_DIMENSION = 3072

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Please set PINECONE_API_KEY and GEMINI_API_KEY environment variables.")

# Initialize clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ─────────────────────────────────────────────
# Fake Patient Discharge Summary Data
# ─────────────────────────────────────────────
# Each entry represents one paragraph from a clinical note,
# exactly as described in the assignment.

SAMPLE_DOCUMENTS = [
    # ── Patient PT-8829 (Dr. Suresh) ──
    {
        "id": "doc-001",
        "text": "Patient PT-8829 was admitted on 12-Jan-2024 with acute chest pain radiating to the left arm. ECG showed ST-segment elevation in leads II, III, and aVF, consistent with an inferior wall myocardial infarction. Troponin-I levels were elevated at 12.5 ng/mL.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2024-01-12"}
    },
    {
        "id": "doc-002",
        "text": "Patient PT-8829 underwent emergency percutaneous coronary intervention (PCI) with stent placement in the right coronary artery. The procedure was successful with TIMI-3 flow restored. Post-procedure ejection fraction was measured at 45%.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2024-01-12"}
    },
    {
        "id": "doc-003",
        "text": "Patient PT-8829 has a history of atrial fibrillation diagnosed in March 2023. Was previously on Amiodarone 200mg daily but switched to Apixaban 5mg twice daily due to recurrent episodes. Echocardiogram from that visit showed mild left atrial enlargement.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2023-03-15"}
    },
    {
        "id": "doc-004",
        "text": "Patient PT-8829 was treated for congestive heart failure (CHF) exacerbation in August 2023. Presented with bilateral pedal edema, orthopnea, and elevated BNP at 1200 pg/mL. Managed with IV Furosemide and fluid restriction. Discharged on Carvedilol 12.5mg and Enalapril 10mg.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2023-08-20"}
    },
    {
        "id": "doc-005",
        "text": "Patient PT-8829 follow-up visit on 10-Feb-2024. Post-MI recovery progressing well. Cardiac rehabilitation initiated. Lipid panel: LDL 95 mg/dL, HDL 38 mg/dL. Started on Rosuvastatin 20mg. Blood pressure well controlled at 128/82 mmHg on current medications.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2024-02-10"}
    },

    # ── Patient PT-1234 (Dr. Suresh) ──
    {
        "id": "doc-006",
        "text": "Patient PT-1234 admitted with uncontrolled Type 2 Diabetes Mellitus. HbA1c measured at 11.2%. Fasting blood glucose 320 mg/dL. Patient reports poor medication compliance and irregular dietary habits. Started on Insulin Glargine 20 units at bedtime.",
        "metadata": {"patient_id": "PT-1234", "doctor_name": "Dr. Suresh", "department": "Endocrinology", "date": "2024-03-05"}
    },
    {
        "id": "doc-007",
        "text": "Patient PT-1234 developed diabetic ketoacidosis (DKA) during hospitalization. Anion gap was 22. Managed with IV insulin drip and aggressive fluid resuscitation. Blood glucose normalized within 18 hours. Transitioned back to subcutaneous insulin regimen.",
        "metadata": {"patient_id": "PT-1234", "doctor_name": "Dr. Suresh", "department": "Endocrinology", "date": "2024-03-06"}
    },

    # ── Patient PT-5567 (Dr. Mehra) ──
    {
        "id": "doc-008",
        "text": "Patient PT-5567 presented with severe headache, neck stiffness, and photophobia. Lumbar puncture performed — CSF analysis showed elevated WBC count (350 cells/µL), low glucose (30 mg/dL), and elevated protein (180 mg/dL), consistent with bacterial meningitis. Started on IV Ceftriaxone and Vancomycin.",
        "metadata": {"patient_id": "PT-5567", "doctor_name": "Dr. Mehra", "department": "Neurology", "date": "2024-04-10"}
    },
    {
        "id": "doc-009",
        "text": "Patient PT-5567 MRI brain showed no abscess or intracranial complications. Blood cultures grew Streptococcus pneumoniae. Antibiotics de-escalated to IV Ceftriaxone alone. Patient showed clinical improvement by day 3 with resolution of fever and neck stiffness.",
        "metadata": {"patient_id": "PT-5567", "doctor_name": "Dr. Mehra", "department": "Neurology", "date": "2024-04-13"}
    },

    # ── Patient PT-8829 additional records ──
    {
        "id": "doc-010",
        "text": "Patient PT-8829 routine lab work from November 2023: Complete blood count within normal limits. Renal function stable with creatinine 1.1 mg/dL. Liver function tests normal. No signs of anemia. Patient advised to continue current cardiac medications.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "General Medicine", "date": "2023-11-15"}
    },
    {
        "id": "doc-011",
        "text": "Patient PT-8829 experienced a transient ischemic attack (TIA) in May 2023. CT angiography revealed 40% stenosis in the left internal carotid artery. Managed conservatively with dual antiplatelet therapy (Aspirin 75mg + Clopidogrel 75mg). No surgical intervention required at this time.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2023-05-22"}
    },
    {
        "id": "doc-012",
        "text": "Patient PT-8829 developed ventricular tachycardia episode during post-MI monitoring in January 2024. Successfully cardioverted with synchronized DC shock at 100J. Started on Amiodarone 400mg loading dose. ICD implantation discussed with patient and family.",
        "metadata": {"patient_id": "PT-8829", "doctor_name": "Dr. Suresh", "department": "Cardiology", "date": "2024-01-14"}
    },
]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Gemini's embedding model."""
    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def main():
    # ── Step 1: Create Pinecone Index ──
    print(f"[INFO] Checking if index '{INDEX_NAME}' exists...")

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"[WARN] Index '{INDEX_NAME}' already exists. Deleting and recreating...")
        pc.delete_index(INDEX_NAME)
        time.sleep(5)

    print(f"[INFO] Creating index '{INDEX_NAME}' with dimension={EMBEDDING_DIMENSION}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Pinecone free tier region
        ),
    )

    # Wait for the index to be ready
    print("[INFO] Waiting for index to be ready...")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("[SUCCESS] Index is ready!")

    # ── Step 2: Connect to the index ──
    index = pc.Index(INDEX_NAME)

    # ── Step 3: Embed all document texts ──
    print(f"[INFO] Embedding {len(SAMPLE_DOCUMENTS)} documents using Gemini...")
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    embeddings = embed_texts(texts)
    print(f"[SUCCESS] Generated {len(embeddings)} embeddings (dimension={len(embeddings[0])})")

    # ── Step 4: Upsert vectors into Pinecone ──
    print("[INFO] Uploading vectors to Pinecone...")
    vectors = []
    for doc, embedding in zip(SAMPLE_DOCUMENTS, embeddings):
        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {
                **doc["metadata"],
                "text": doc["text"],  # Store original text in metadata for retrieval
            },
        })

    index.upsert(vectors=vectors)
    print(f"[SUCCESS] Successfully uploaded {len(vectors)} vectors to Pinecone!")

    # ── Step 5: Verify ──
    time.sleep(3)  # Wait for indexing
    stats = index.describe_index_stats()
    print(f"\n[INFO] Index Stats: {stats.total_vector_count} vectors stored")
    print("\n[SUCCESS] Setup complete! You can now run app.py")


if __name__ == "__main__":
    main()
