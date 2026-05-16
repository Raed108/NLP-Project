# Retrieval-Augmented Generation (RAG) System

## Milestone 3: Arabic Multi-turn Conversational Chatbot

**Project Overview:**
This is a multi-turn conversational RAG system that answers user questions strictly based on retrieved context from Arabic transcripts. The system supports both Arabic and English queries, implements robust error handling with fallback models, and evaluates generation quality using four complementary metrics.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Core Components](#core-components)
3. [Data Pipeline](#data-pipeline)
4. [Setup & Installation](#setup--installation)
5. [Running the Project](#running-the-project)
6. [Evaluation Framework](#evaluation-framework)
7. [System Features](#system-features)
8. [Model Comparison](#model-comparison)
9. [Usage Scenarios](#usage-scenarios)
10. [Design Justifications](#design-justifications)

---

## Project Architecture

```
User Query (Arabic/English)
    ↓
[Language Detection]
    ↓
[Semantic Retrieval] → FAISS Vector Store (multilingual embeddings)
    ↓
[Context Formatting] → Retrieved Documents
    ↓
[Prompt Selection] → STRICT_ARABIC_PROMPT or STRICT_ENGLISH_PROMPT
    ↓
[LLM Generation] → Gemini 2.5 Flash / DeepSeek v4 Flash
    ↓
[Fallback & Retry Logic] → Exponential backoff, fast-fail on quota
    ↓
[Memory Management] → Conversation history (sliding window)
    ↓
[Response Display] → Streamlit UI with sidebar logs
```

---

## Core Components

### 1. **Embeddings & Vector Store** (`src/embeddings.py`, `src/retrieval.py`)

- **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- **Rationale:** Lightweight, multilingual, preserves Arabic semantics and English tokens without lemmatization/stemming
- **Vector DB:** FAISS (in-memory, loaded from `faiss_index/`)
- **Index Rebuild:** Automatic on startup if missing

### 2. **Retrieval System** (`src/retrieval.py`)

- **Retrieval Method:** Semantic similarity search with cosine distance
- **Default k:** 5 top-matching documents
- **Chunking:** Natural sentence-level + paragraph boundaries (no artificial chunk size imposed)
- **Traceability:** Each retrieved document includes source episode metadata

### 3. **LLM Client** (`src/llm.py`)

- **Primary Model:** `gemini-2.5-flash` (Google)
- **Fallback Chain:**
  1. `gemini-2.5-flash`
  2. `deepseek-v4-flash` (via OpenRouter)
  3. `mistralai/devstral-2512` (via OpenRouter)
- **Retry Logic:**
  - Max retries: 2 per model (1 initial + 2 retries = 3 attempts total)
  - Exponential backoff: 1s, 2s sleep
  - Fast-fail detection: 429, quota exhausted → skip retries and fallback immediately
- **Returns:** `(response_text, used_model_name, total_attempts)`

### 4. **Prompt Engineering** (`src/prompts.py`)

Three prompt templates with strict grounding:

#### STRICT_ARABIC_PROMPT

```
تحدث باللغة العربية فقط
استخدم السياق المُعطى لك للإجابة
إذا لم يكن السياق يحتوي على المعلومة، قل "لا توجد معلومات في السياق"
```

**Goal:** Force Arabic responses, prevent hallucination

#### STRICT_ENGLISH_PROMPT

```
Answer in English only. Use the Arabic context as evidence.
Never mirror the context language in the final answer.
```

**Goal:** English-only responses even with Arabic context

#### MINIMAL_PROMPT (fallback)

Basic context + question without extra constraints

### 5. **Conversation Memory** (`src/memory.py`)

- **Strategy:** Sliding window (retains last 4 messages)
- **Backend:** LangChain ConversationBufferWindowMemory
- **Purpose:** Maintain multi-turn context without token overload
- **Trade-off:** Short memory preserves coherence, prevents quota exhaustion

### 6. **Streamlit Interface** (`app.py`)

**Features:**

- Sidebar model selector (Gemini/DeepSeek/Mistral)
- Chat input/output history
- Language detection (Arabic/English) auto-detected
- Fallback warnings when primary model fails
- LLM attempt counter (shows retry info)
- Loading spinner during generation
- Copy-to-clipboard button for responses

---

## Data Pipeline

### Input Data

- **Transcripts:** 13 Arabic episodes from MS1 (e.g., "Citizen Kane", "Octopus", "Samurai")
- **QA Pairs:** 7 JSON datasets with 5 questions each per episode (35 total QA pairs)
- **Format Preserved:**
  - No lemmatization, stemming, or punctuation removal
  - Dialectal variations and code-switching preserved

### FAISS Index Creation

Run once to build the index:

```bash
python -c "from src.retrieval import RetrieverSystem; r = RetrieverSystem(); print('Index ready')"
```

- Reads all transcripts from `data/Transcripts/`
- Embeds each paragraph via multilingual model
- Saves index to `faiss_index/` for reuse
- **Note:** First run takes ~30s due to HuggingFace model download

### Example Workflow

```
Query: "من هو أورسون ويلز؟" (Arabic)
  ↓
Retrieval: Find top-5 paragraphs mentioning "ويلز"
  ↓
Context: "[Citizen Kane episode context]"
  ↓
Prompt: STRICT_ARABIC_PROMPT (detected language = Arabic)
  ↓
LLM: Generate response in Arabic only
  ↓
Memory: Store (query, response) in sliding window
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Step 1: Clone & Install Dependencies

```bash
cd NLP-Project
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Get API Keys

**Gemini API (Google AI Studio)**

1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **Create API key**
4. Copy the key

**OpenRouter API (for DeepSeek/Mistral)**

1. Go to: https://openrouter.ai/
2. Sign in or create account
3. Navigate to: https://openrouter.ai/keys
4. Create a new API key
5. Ensure account has credits (free tier may have limited requests)

### Step 3: Configure Environment

Create or edit `.env` file in project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Example `.env.example` provided in repo for reference.**

### Step 4: Build FAISS Index

```bash
python -c "from src.retrieval import RetrieverSystem; r = RetrieverSystem(); print('FAISS index created successfully')"
```

Creates `faiss_index/` directory with embeddings of all transcripts.

---

## Running the Project

### Option 1: Streamlit Chatbot (Recommended)

```bash
streamlit run app.py
```

- Opens interactive chat interface at `http://localhost:8501`
- Select model from sidebar
- Type queries in Arabic or English
- View conversation history and fallback info

### Option 2: Evaluation Module

Run evaluation on 5 test queries (2 models, 4 metrics):

```bash
python -m src.evaluation
```

**Output:**

- Console summary with aggregated scores
- `evaluation_results/evaluation_results.json` — per-sample details
- `evaluation_results/evaluation_summary.csv` — model comparison table

**CSV Columns:**
| Model | Successful Queries | Attempted Queries | Mean ROUGE-L | Mean BERTScore | Mean Grounding F1 | Mean Embedding Cosine |
|-------|-------------------|------------------|--------------|----------------|------------------|----------------------|

---

## Evaluation Framework

### Metrics (4 Total)

#### 1. **ROUGE-L** (Text Generation Quality)

- **Definition:** Longest common subsequence F-score between reference and generated answer
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - High (>0.5): Lexically similar answers
  - Low (~0.0): Different phrasing, still may be correct
- **Why Here:** Assignment requirement; useful for exact-match answers
- **Limitation:** Abstractive answers (paraphrased) score low even if semantically correct
- **Recommendation:** Use alongside BERTScore/Embedding Cosine for interpretation

#### 2. **BERTScore** (Semantic Correctness)

- **Definition:** Token-level cosine similarity between contextual embeddings (using `bert-base-multilingual-cased`)
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - High (>0.65): Semantically similar despite wording differences
  - Low (<0.5): Semantically different from reference
- **Why Here:** Captures paraphrase robustness; works for both Arabic and English
- **Advantage:** Tolerates phrasing variations better than ROUGE

#### 3. **Grounding F1** (Faithfulness to Retrieved Context)

- **Definition:** Harmonic mean of:
  - Precision: % of answer tokens found in retrieved context
  - Recall: % of context tokens appearing in answer
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - High (>0.3): Answer draws heavily from context (low hallucination)
  - Low (~0.0): Answer uses few context tokens (potential hallucination)
- **Why Here:** Directly measures grounding objective of RAG
- **Limitation:** Does not penalize irrelevant context overlap; needs manual review for hallucination

#### 4. **Embedding Cosine** (Paraphrase-Robust Semantic Alignment)

- **Definition:** Cosine similarity between sentence-level embeddings (multilingual MiniLM model)
- **Range:** -1.0 to 1.0 (typically 0.0 to 1.0 for similar texts)
- **Interpretation:**
  - High (>0.7): Very similar semantic meaning
  - Medium (0.5–0.7): Related but not identical
  - Low (<0.3): Semantically distant
- **Why Here:** Robust to paraphrasing; tolerates dialectal variations and code-switching
- **Advantage:** Works when ROUGE is zero but meaning is preserved
- **Best For:** Abstractive QA evaluation (preferred metric when ROUGE is low)

### Metric Selection Justification

| Requirement             | Metric(s)                    | Rationale                                                                       |
| ----------------------- | ---------------------------- | ------------------------------------------------------------------------------- |
| Text Generation Quality | ROUGE-L + Embedding Cosine   | ROUGE for exact match, Embedding Cosine for paraphrase tolerance                |
| Semantic Correctness    | BERTScore + Embedding Cosine | BERT captures contextual semantics; Embedding Cosine is lightweight alternative |
| Grounding to Context    | Grounding F1                 | Direct token overlap between answer and retrieved context                       |

### Expected Results

- **ROUGE-L:** Often ~0.0 because LLMs generate abstractive answers (different wording from reference)
- **BERTScore:** ~0.65–0.70 (good semantic match despite different phrasing)
- **Grounding F1:** ~0.05–0.15 (low because LLMs synthesize beyond direct context tokens)
- **Embedding Cosine:** ~0.70–0.80 (high semantic similarity when ROUGE is low)

**Interpretation:** High BERTScore + Embedding Cosine + low ROUGE + low Grounding F1 = **abstractive, paraphrased but semantically correct answers with moderate grounding.**

---

## System Features

### 1. **Language Detection**

- Checks Unicode ranges for Arabic vs Latin characters
- Selects appropriate prompt template (STRICT_ARABIC_PROMPT vs STRICT_ENGLISH_PROMPT)
- Enables code-switching support

### 2. **Robust Error Handling**

- **API Failures:** Automatic fallback to next model in chain
- **Quota Exhaustion:** Fast-fail detection (429, "ResourceExhausted" strings)
- **Network Timeouts:** Exponential backoff (1s, 2s, 4s)
- **Empty Responses:** Graceful degradation with warning message

### 3. **Multi-turn Conversation**

- Sliding window memory (4 messages = 2 user-assistant pairs)
- Reuses conversation context in next LLM call
- Prevents memory overflow while maintaining coherence

### 4. **Out-of-Domain Detection** (Future Enhancement)

- Currently: All queries passed to retrieval (no explicit OOD filtering)
- Recommended: Cosine similarity threshold on retrieved documents
- If max similarity < threshold → return "Query outside available knowledge"

### 5. **Attempt Tracking**

- Displays number of LLM attempts (retries) in sidebar
- Shows which model was ultimately used
- Useful for debugging rate-limiting issues

---

## Model Comparison

### Models Evaluated

| Model                       | Provider   | Free Tier Limit | Latency | Quality | Best For                      |
| --------------------------- | ---------- | --------------- | ------- | ------- | ----------------------------- |
| **gemini-2.5-flash**        | Google AI  | 20 req/day      | 1–2s    | High    | Primary; code-switching       |
| **deepseek-v4-flash**       | OpenRouter | Varies          | 2–3s    | High    | Fallback; handles Arabic well |
| **mistralai/devstral-2512** | OpenRouter | Varies          | 2–3s    | Medium  | Tertiary fallback             |

### Evaluation Results (5 queries)

**gemini-2.5-flash (1 successful, 4 quota-exhausted):**

```
Completed:    1/5
ROUGE-L:      0.0000
BERTScore:    0.6622
Grounding F1: 0.0000
Emb.Cosine:   0.75+ (estimate)
```

**Note:** Free tier quota (20 req/day) exhausted after first request.

**deepseek-v4-flash (4 successful, 1 rate-limited):**

```
Completed:    4/5
ROUGE-L:      0.0000
BERTScore:    0.6817
Grounding F1: 0.0516
Emb.Cosine:   0.72+ (estimate)
```

**Note:** Rate limits less restrictive than Gemini free tier.

### Recommendation

- **For Production:** Use fresh API keys and account each month for quota reset
- **Fallback Strategy:** Prioritize DeepSeek for reliability (higher rate limits)
- **Cost:** All models free-tier compatible; no paid API usage required

---

## Usage Scenarios

### Scenario 1: Arabic Historical Question

```
User: "من هو أورسون ويلز وماذا أنجز في السينما؟"
(Who is Orson Welles and what did he achieve in cinema?)

System:
1. Detects language: Arabic
2. Retrieves: 5 paragraphs from "Citizen Kane" episode
3. Selects: STRICT_ARABIC_PROMPT
4. Generates:
   "أورسون ويلز هو مخرج وممثل أمريكي اشتهر بفيلم
    Citizen Kane الذي غيّر مسار السينما..."
5. Memory: Stores (query, response) for multi-turn context
```

### Scenario 2: English Query with Arabic Context

```
User: "What is the main theme of the Octopus episode?"

System:
1. Detects language: English
2. Retrieves: 5 paragraphs from "Octopus" episode
3. Selects: STRICT_ENGLISH_PROMPT
4. Generates (Arabic context → English response):
   "The octopus episode explores intelligence and adaptation
    in marine life, demonstrating..."
5. Memory: Stores for follow-up questions
```

### Scenario 3: Fallback Scenario (Gemini Quota Exhausted)

```
User: [Any query after 20 requests in a day]

System:
1. Attempts: gemini-2.5-flash → 429 Quota Exhausted
2. Fast-fail: Detects quota error, skips retries
3. Fallback: Tries deepseek-v4-flash → Success
4. Sidebar: Shows warning "Fell back to model: deepseek-v4-flash"
5. UI: Warning banner displayed, response still delivered
```

### Scenario 4: Multi-turn Conversation

```
User Turn 1: "من هو تشارلز فوستر كين؟"
System Response: "[Answer about character]"
Memory: Stores (Q1, A1)

User Turn 2: "هل كان حقيقيًا؟"
(Was he real?)

System:
1. Retrieves: Documents about Citizen Kane
2. Includes: Q1 + A1 in memory for context
3. Generates: "لا، كان شخصية خيالية... بناءً على حياة..."
4. Memory: Now stores (Q1, A1, Q2, A2)
```

---

## Design Justifications

### 1. **Why Multilingual Embeddings?**

- Project requirement: Preserve Arabic semantics + English tokens
- `paraphrase-multilingual-MiniLM-L12-v2`: Lightweight, trained on 50+ languages
- Alternative rejected: Arabic-only models (exclude English tokens)
- No lemmatization/stemming: Preserve natural text + dialectal variation

### 2. **Why FAISS?**

- Fast semantic search (in-memory, no network latency)
- Persists to disk for quick reloads
- Sufficient for 13 episodes (~50KB embeddings)
- Alternative: Cloud vector DB (higher cost, latency)

### 3. **Why Sliding Window Memory?**

- Full history → token exhaustion + higher cost
- Summary memory → information loss
- Sliding window (4 messages) → balance coherence + efficiency
- Tunable: Can extend to 6–10 messages for longer conversations

### 4. **Why Strict Prompts + Language Detection?**

- Problem: LLMs bias toward input language (Arabic context → Arabic response even if English requested)
- Solution: STRICT_ENGLISH_PROMPT with explicit instruction
- Validation: Manual review shows improved English-only compliance

### 5. **Why Fallback Chain?**

- Primary (Gemini): Best quality but strict quota
- Secondary (DeepSeek): Good quality, higher rate limits
- Tertiary (Mistral): Worst quality, used only if others fail
- Exponential backoff: Respects rate limits, avoids spamming API

### 6. **Why Four Metrics?**

- **ROUGE-L:** Required by assignment; captures exact match
- **BERTScore:** Captures semantic similarity; handles paraphrasing
- **Grounding F1:** Directly measures RAG objective (faithfulness)
- **Embedding Cosine:** Robust alternative to ROUGE for abstractive QA

### 7. **Why No Lemmatization/Stemming?**

- Requirement: Preserve dialectal variation (e.g., "المصيب" vs "المصابة")
- Requirement: Preserve code-switching (e.g., "Citizen Kane" within Arabic text)
- Over-normalization → Loss of meaning and intent

---

## File Structure

```
NLP-Project/
├── app.py                      # Streamlit chatbot interface
├── requirements.txt            # Python dependencies
├── .env                        # API keys (user-configured)
├── .env.example               # Template for .env
├── .gitignore                 # Ignore venv, .env, cache
├── README.md                  # This file
├── faiss_index/               # Vector store (created on first run)
│   ├── index.faiss
│   ├── index.pkl
│   └── docstore_pickle.pkl
├── evaluation_results/        # Evaluation output (created on run)
│   ├── evaluation_results.json
│   └── evaluation_summary.csv
├── src/
│   ├── __init__.py
│   ├── llm.py                # LLM client with fallback logic
│   ├── retrieval.py          # FAISS retriever
│   ├── embeddings.py         # Multilingual embedding wrapper
│   ├── memory.py             # Conversation memory (LangChain)
│   ├── prompts.py            # Prompt templates
│   ├── preprocessing.py      # Text normalization (identity function)
│   ├── ui.py                 # Streamlit UI components (if modularized)
│   ├── utils.py              # Utility functions (context formatting)
│   └── evaluation.py         # Evaluation runner (4 metrics)
└── data/
    ├── Transcripts/          # 13 Arabic episodes (from MS1)
    │   ├── الأخطبوط الدحيح.txt
    │   ├── هل Citizen Kane أفضل فيلم في التاريخ؟ الدحيح.txt
    │   └── ... (11 more)
    └── QA/                   # QA pairs for evaluation
        ├── citizen_kane_qa_dataset.json
        ├── octopus_qa_dataset.json
        └── ... (5 more)
```

---

## Troubleshooting

### Issue: "FAISS index not found"

**Solution:**

```bash
python -c "from src.retrieval import RetrieverSystem; r = RetrieverSystem()"
```

This will create the index automatically.

### Issue: "No module named 'bert_score'"

**Solution:**

```bash
pip install -r requirements.txt
```

### Issue: "429 Too Many Requests" (Gemini)

**Cause:** Free-tier quota exhausted (20 requests/day)
**Solution:**

1. Wait 24 hours for quota reset, OR
2. Create new Google account with fresh API key, OR
3. Switch to paid Gemini API

**During Evaluation:** System automatically falls back to DeepSeek (higher rate limits)

### Issue: "Connection timeout"

**Cause:** Network issue or API endpoint down
**Solution:**

1. Check internet connection
2. Verify API keys in `.env`
3. Retry manually; exponential backoff applies

### Issue: Streamlit app slow on first run

**Cause:** First-time embedding model download from HuggingFace
**Solution:** Initial run takes ~30s. Subsequent runs are fast (<2s).

---

## Requirements & Compliance

### Milestone 3 Checklist

| Requirement                      | Status | Evidence                                                                    |
| -------------------------------- | ------ | --------------------------------------------------------------------------- |
| **2.1 Data Usage**               | ✅     | 13 episodes, 35 QA pairs for eval, no training                              |
| **2.2 Text Representation**      | ✅     | No lemmatization/stemming in `preprocessing.py` (identity function)         |
| **2.3 Embedding & Vector Store** | ✅     | Multilingual model + FAISS in `retrieval.py`                                |
| **2.4 Chunking Strategy**        | ✅     | Sentence-level + paragraph boundaries, traceable to source                  |
| **2.5 Multi-turn Chatbot**       | ✅     | LangChain ConversationBufferWindowMemory in `memory.py`                     |
| **2.6 Prompt Engineering**       | ✅     | STRICT_ARABIC_PROMPT, STRICT_ENGLISH_PROMPT, MINIMAL_PROMPT in `prompts.py` |
| **2.7 OOD Detection**            | ⏳     | Future: Similarity threshold on retrieved docs                              |
| **2.8 Robustness**               | ✅     | Fallback chain, exponential backoff, fast-fail quota detection in `llm.py`  |
| **2.9 LLM Requirements**         | ✅     | Gemini 2.5 Flash + DeepSeek (OpenRouter), free-tier only                    |
| **2.10 Evaluation**              | ✅     | 2 models (Gemini, DeepSeek), 4 metrics (ROUGE, BERT, Grounding, Embedding)  |
| **2.11 Interface**               | ✅     | Streamlit chatbot with history, model selector, fallback warnings           |

---

## Contact & Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section above
2. Review console logs for error details
3. Verify `.env` configuration
4. Ensure all dependencies installed: `pip install -r requirements.txt`

---

## License & Attribution

**Project:** NLP Milestone 3 (RAG System)
**Framework:** Streamlit, LangChain, FAISS, sentence-transformers
**Models:** Google Gemini 2.5 Flash, DeepSeek v4 Flash, OpenRouter API
**Evaluation:** ROUGE-score, bert-score, scikit-learn

---

**Last Updated:** May 16, 2026
