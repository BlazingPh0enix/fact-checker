from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import spacy
import os
import json
from pathlib import Path
from openai import OpenAI
import re
from dotenv import load_dotenv
import pickle
import hashlib

# Load environment variables
load_dotenv()

# Initialize the SentenceTransformer model with error handling
def load_sentence_transformer():
    """Load the sentence transformer model with error handling"""
    try:
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to download model fresh...")
        try:
            # Force download with trust_remote_code=True
            model = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=True)
            print("Model downloaded and loaded successfully!")
            return model
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise e2

# Global model variable - will be loaded when needed
model = None

def embed_headlines(headlines):
    """
    Embed headlines using a pre-trained SentenceTransformer model.
    """
    global model
    if model is None:
        model = load_sentence_transformer()
    
    print("Embedding headlines...")
    embeddings = model.encode(headlines.tolist(), show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    """
    Create a FAISS index from the embeddings.
    """
    print("Creating FAISS index...")
    dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(np.array(embeddings).astype(np.float32))  # type: ignore # Add embeddings to the index
    return index

def claim_extractor():
    """
    Initialize and return a spaCy claim extractor.
    Always includes the full input text as the first claim.
    """
    print("Loading spaCy model for claim extraction...")
    nlp = spacy.load("en_core_web_sm")
    def extract_claims(text):
        claims = [text.strip()]  # Always include the full input text as the first claim
        doc = nlp(text)
        # Extract noun chunks and named entities as potential claims
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text) > 3 and chunk_text.lower() != text.strip().lower():
                claims.append(chunk_text)
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if ent.label_ in ["ORG", "GPE", "PERSON", "EVENT", "LAW", "MONEY"] and ent_text.lower() != text.strip().lower():
                claims.append(ent_text)
        # If no good claims found, add all sentences (except duplicates)
        if len(claims) == 1:
            claims.extend([sent.text.strip() for sent in doc.sents if sent.text.strip().lower() != text.strip().lower()])
        # Remove duplicates, preserve order
        seen = set()
        unique_claims = []
        for c in claims:
            c_norm = c.lower()
            if c_norm not in seen:
                seen.add(c_norm)
                unique_claims.append(c)
        return unique_claims
    return extract_claims

def llm_verdict(claim, facts):
    """
    Use OpenAI's GPT to analyze claims and return verdicts.
    """
    print(f"Analyzing claim with OpenAI: {claim}")
    prompt = f"""
    Given the claim: "{claim}"
    And the following verified facts from trusted sources:
    {chr(10).join([f"- {fact}" for fact in facts])}

    Classify the claim as one of:
    - True: The claim is supported by the provided facts
    - False: The claim contradicts the provided facts  
    - Unverifiable: There's insufficient information in the facts to verify the claim

    Provide your response in JSON format:
    {{
        "verdict": "True/False/Unverifiable",
        "evidence": ["relevant fact 1", "relevant fact 2"],
        "reasoning": "Brief explanation of why you reached this conclusion"
    }}
    """
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant. Analyze claims against provided facts and respond only in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        result_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        if result_text:
            try:
                # Find JSON in the response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "verdict": "Unverifiable",
                    "evidence": facts[:2] if facts else [],
                    "reasoning": "Unable to parse LLM response properly"
                }
        else:
            result = {
                "verdict": "Unverifiable", 
                "evidence": facts[:2] if facts else [],
                "reasoning": "Empty response from LLM"
            }
        
        return result
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": f"Error occurred during analysis: {str(e)}"
        }

def load_fact_database(csv_path="pib_headlines.csv"):
    """
    Load the fact database from CSV file.
    """
    print(f"Loading fact database from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Use headline column as facts
        facts = df['headline'].dropna().tolist()
        print(f"Loaded {len(facts)} facts from database")
        return facts
    except Exception as e:
        print(f"Error loading fact database: {e}")
        return []

def retrieve_similar_facts(query, index, facts, k=5):
    """
    Retrieve top-k similar facts from the FAISS index.
    """
    global model
    if model is None:
        model = load_sentence_transformer()
        
    print(f"Retrieving similar facts for: {query}")
    
    # Embed the query
    query_embedding = model.encode([query])
    
    # Search the index
    distances, indices = index.search(np.array(query_embedding).astype(np.float32), k)
    
    # Get the corresponding facts
    retrieved_facts = []
    for i, idx in enumerate(indices[0]):
        if idx < len(facts):  # Ensure index is valid
            retrieved_facts.append({
                'fact': facts[idx],
                'distance': distances[0][i],
                'similarity': 1 / (1 + distances[0][i])  # Convert distance to similarity
            })
    
    return retrieved_facts

def get_file_hash(filepath):
    """Return a hash of the file contents for cache key."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_or_create_embeddings_and_index(facts, facts_csv):
    """Load or create embeddings and FAISS index, using cache if available."""
    # Use hash of the fact database file as cache key
    file_hash = get_file_hash(facts_csv)
    CACHE_DIR = Path("cache")
    CACHE_DIR.mkdir(exist_ok=True)
    emb_path = CACHE_DIR / f"embeddings_{file_hash}.pkl"
    index_path = CACHE_DIR / f"faiss_index_{file_hash}.index"

    if emb_path.exists() and index_path.exists():
        print("Loading embeddings and FAISS index from cache...")
        with open(emb_path, 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(str(index_path))
        return embeddings, index
    else:
        print("Cache not found. Creating embeddings and FAISS index...")
        embeddings = embed_headlines(pd.Series(facts))
        index = create_faiss_index(embeddings)
        with open(emb_path, 'wb') as f:
            pickle.dump(embeddings, f)
        faiss.write_index(index, str(index_path))
        return embeddings, index

def fact_check_pipeline(input_text, facts_csv="pib_headlines.csv"):
    """
    Main fact-checking pipeline that ties everything together.
    """
    print("=" * 50)
    print("FACT-CHECKING PIPELINE STARTED")
    print("=" * 50)
    
    # Load fact database
    facts = load_fact_database(facts_csv)
    if not facts:
        return {
            "error": "Could not load fact database",
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": "Fact database is empty or could not be loaded"
        }

    # Check for exact match (case-insensitive, strip)
    facts_normalized = [f.strip().lower() for f in facts]
    input_normalized = input_text.strip().lower()
    if input_normalized in facts_normalized:
        idx = facts_normalized.index(input_normalized)
        return {
            "claim": input_text,
            "verdict": "True",
            "evidence": [facts[idx]],
            "reasoning": "The input exactly matches a verified fact in the database.",
            "similar_facts": [{"fact": facts[idx], "distance": 0.0, "similarity": 1.0}],
            "confidence": 1.0
        }

    #Load or create embeddings and FAISS index (with cache)
    embeddings, index = load_or_create_embeddings_and_index(facts, facts_csv)
    
    #Extract claims from input text
    extract_claims = claim_extractor()
    claims = extract_claims(input_text)
    print(f"Extracted claims: {claims}")

    #Check for exact match for any claim
    for claim in claims:
        claim_normalized = claim.strip().lower()
        if claim_normalized in facts_normalized:
            idx = facts_normalized.index(claim_normalized)
            return {
                "claim": claim,
                "verdict": "True",
                "evidence": [facts[idx]],
                "reasoning": "The claim exactly matches a verified fact in the database.",
                "similar_facts": [{"fact": facts[idx], "distance": 0.0, "similarity": 1.0}],
                "confidence": 1.0
            }
    
    #For each claim, retrieve similar facts and get LLM verdict
    results = []
    
    for claim in claims:
        print(f"\nProcessing claim: {claim}")
        
        # Retrieve similar facts
        similar_facts = retrieve_similar_facts(claim, index, facts, k=5)
        
        if not similar_facts:
            claim_result = {
                "claim": claim,
                "verdict": "Unverifiable",
                "evidence": [],
                "reasoning": "No similar facts found in database"
            }
        else:
            # Extract just the fact text for LLM analysis
            fact_texts = [item['fact'] for item in similar_facts]
            
            # Get LLM verdict
            llm_result = llm_verdict(claim, fact_texts)
            
            claim_result = {
                "claim": claim,
                "verdict": llm_result.get("verdict", "Unverifiable"),
                "evidence": llm_result.get("evidence", fact_texts[:2]),
                "reasoning": llm_result.get("reasoning", "No reasoning provided"),
                "similar_facts": similar_facts,
                "confidence": sum(item['similarity'] for item in similar_facts) / len(similar_facts) if similar_facts else 0
            }
        
        results.append(claim_result)
    
    #Aggregate results (if multiple claims)
    if len(results) == 1:
        final_result = results[0]
    else:
        # Simple aggregation: if any claim is False, overall is False
        # If all are True, overall is True, otherwise Unverifiable
        verdicts = [r["verdict"] for r in results]
        if "False" in verdicts:
            overall_verdict = "False"
        elif all(v == "True" for v in verdicts):
            overall_verdict = "True"
        else:
            overall_verdict = "Unverifiable"
            
        final_result = {
            "overall_verdict": overall_verdict,
            "individual_claims": results,
            "input_text": input_text
        }
    
    print("=" * 50)
    print("FACT-CHECKING PIPELINE COMPLETED")
    print("=" * 50)
    
    return final_result

# Example usage and testing
if __name__ == "__main__":
    # Test the pipeline
    test_input = "The Indian government has announced free electricity to all farmers starting July 2025."
    result = fact_check_pipeline(test_input)
    print("\nFINAL RESULT:")
    print(json.dumps(result, indent=2))