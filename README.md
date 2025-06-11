# Truth Checker - LLM-Powered Fact Verification System

A lightweight fact-checking system that analyzes news posts and social media statements using Retrieval-Augmented Generation (RAG) with custom embeddings and robust exact-match logic.

## Features

- **🔍 Claim Extraction**: Uses spaCy NLP to extract key claims and entities from input text, always including the full input as a claim
- **⚡ Exact Match Detection**: Instantly returns 'True' if your input or any claim exactly matches a fact in the database (case-insensitive)
- **📊 Vector Database**: Embeddings stored in FAISS for fast similarity search, with persistent disk caching
- **🤖 LLM Analysis**: OpenAI GPT for intelligent claim verification
- **🌐 Web Interface**: Beautiful Streamlit app for easy interaction
- **📈 Confidence Scoring**: Similarity-based confidence metrics

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv env
# Windows:
env\Scripts\activate
# Mac/Linux:
# source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure API Key

Set your OpenAI API key as an environment variable (recommended):

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Mac/Linux
export OPENAI_API_KEY=your_api_key_here
```

Or add it to a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Application

#### Streamlit Web App (Recommended)
```bash
streamlit run app.py
```

#### Command Line
```bash
python main.py
```

## How It Works

1. **Input Processing**: Full input is always checked for exact match in the fact database
2. **Claim Extraction**: spaCy NLP extracts claims/entities, always including the full input
3. **Embedding**: Convert text to vectors using Sentence Transformers
4. **Retrieval**: Find similar facts in FAISS vector database (with persistent disk caching)
5. **Analysis**: LLM compares claims against retrieved facts
6. **Verdict**: Classify as True ✅, False ❌, or Unverifiable 🤷‍♂️

## System Architecture

```
Input Text → [Exact Match Check] → Claim Extraction → Embedding → Vector Search → LLM Analysis → Verdict
              (fast)                (spaCy)           (SentenceT)   (FAISS)      (OpenAI)     (JSON)
```

## Example Usage

### Input
```
Prime Minister greets the people of Telangana on their Statehood Day
```

### Output (if present in database)
```json
{
  "claim": "Prime Minister greets the people of Telangana on their Statehood Day",
  "verdict": "True",
  "evidence": ["Prime Minister greets the people of Telangana on their Statehood Day"],
  "reasoning": "The input exactly matches a verified fact in the database.",
  "similar_facts": [{"fact": "Prime Minister greets the people of Telangana on their Statehood Day", "distance": 0.0, "similarity": 1.0}],
  "confidence": 1.0
}
```

## File Structure

```
├── main.py              # Core fact-checking pipeline
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
├── pib_headlines.csv    # Fact database (PIB press releases)
├── README.md            # This file
├── cache/               # Embedding and FAISS index cache
└── env/                 # Virtual environment
```

## Key Components

### 1. Claim Extraction (`claim_extractor()`)
- Uses spaCy's NLP pipeline
- **Always includes the full input text as a claim**
- Extracts noun chunks and named entities
- Filters for meaningful claims

### 2. Exact Match Logic
- Checks for exact (case-insensitive, whitespace-trimmed) match of input or any claim against the fact database
- Returns 'True' instantly if found, bypassing embedding/LLM

### 3. Embedding System (`embed_headlines()`)
- Sentence Transformers model: `all-MiniLM-L6-v2`
- Creates dense vector representations
- Optimized for semantic similarity
- **Embeddings are cached on disk for fast startup**

### 4. Vector Database (`create_faiss_index()`)
- FAISS IndexFlatL2 for L2 distance search
- Fast similarity retrieval
- **Index is cached on disk for fast startup**

### 5. LLM Integration (`llm_verdict()`)
- OpenAI GPT-4.1 nano for reasoning
- Structured JSON output
- Error handling and fallbacks

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **OpenAI API errors**
   - Check API key is valid
   - Ensure you have credits
   - Check internet connection

3. **FAISS installation issues**
   ```bash
   pip install faiss-cpu  # for CPU-only version
   ```

4. **Cache not updating after changing facts**
   - The cache is keyed by a hash of the fact database file. If you update `pib_headlines.csv`, the cache will refresh automatically on next run.
   - If you want to force a refresh, delete the files in the `cache/` directory.

5. **Exact match not detected**
   - Ensure your input matches a fact in the database exactly (case-insensitive, ignoring leading/trailing whitespace).
   - The full input is always checked first, then each extracted claim.

6. **Memory issues**
   - Reduce batch size in embedding
   - Use smaller models
   - Process fewer facts at once
