import os
import torch
import hashlib

# Set environment variables early to avoid PyTorch conflicts
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

torch.classes.__path__ = []

# Set page config
st.set_page_config(
    page_title="Truth Checker - LLM-Powered Fact Verification", 
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling and text visibility
st.markdown("""
<style>
    /* Fix text visibility issues */
    .main .block-container {
        color: #262730 !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .verdict-true {
        color: #28a745 !important;
        font-weight: bold;
        font-size: 1.5rem;
        background-color: #d4edda;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 2px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .verdict-false {
        color: #dc3545 !important;
        font-weight: bold;
        font-size: 1.5rem;
        background-color: #f8d7da;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 2px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .verdict-unverifiable {
        color: #856404 !important;
        font-weight: bold;
        font-size: 1.5rem;
        background-color: #fff3cd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 2px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .claim-box {
        background-color: #f8f9fa;
        color: #495057 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .evidence-box {
        background-color: #e9ecef;
        color: #495057 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #ced4da;
    }
    
    /* Ensure all text is visible - more specific selectors */
    .stMarkdown p, .stMarkdown div, .stMarkdown span, 
    .stMarkdown li, .stMarkdown h1, .stMarkdown h2, 
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #262730 !important;
    }
    
    /* Streamlit components */
    .stText {
        color: #262730 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg, .css-1v0mbdj {
        color: #262730 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Import main functionality after environment setup
try:
    from main import fact_check_pipeline
    import_success = True
except Exception as e:
    import_success = False
    import_error = str(e)

def display_verdict_emoji(verdict):
    """Return appropriate emoji for verdict"""
    if verdict.lower() == "true":
        return "✅"
    elif verdict.lower() == "false":
        return "❌"
    else:
        return "🤷‍♂️"

def display_verdict_style(verdict):
    """Return appropriate CSS class for verdict"""
    if verdict.lower() == "true":
        return "verdict-true"
    elif verdict.lower() == "false":
        return "verdict-false"
    else:
        return "verdict-unverifiable"

def display_claim_result(claim_result):
    """Display results for a single claim"""
    verdict = claim_result.get('verdict', 'Unverifiable')
    emoji = display_verdict_emoji(verdict)
    css_class = display_verdict_style(verdict)
    
    # Verdict with better styling
    st.markdown(f'<div class="{css_class}">{emoji} <strong>{verdict}</strong></div>', unsafe_allow_html=True)
    
    # Reasoning
    if 'reasoning' in claim_result and claim_result['reasoning']:
        st.markdown("**🧠 Reasoning:**")
        st.markdown(f'<div class="claim-box">{claim_result["reasoning"]}</div>', unsafe_allow_html=True)
    
    # Evidence
    if 'evidence' in claim_result and claim_result['evidence']:
        st.markdown("**📚 Supporting Evidence:**")
        for i, evidence in enumerate(claim_result['evidence'], 1):
            if evidence and evidence.strip():  # Only show non-empty evidence
                st.markdown(f'<div class="evidence-box"><strong>Evidence {i}:</strong> {evidence}</div>', 
                           unsafe_allow_html=True)
      # Similar facts (if available) - Display directly without nested expander
    if 'similar_facts' in claim_result and claim_result['similar_facts']:
        st.markdown("**🔍 Similar Facts from Database:**")
        for i, fact_info in enumerate(claim_result['similar_facts'][:3], 1):  # Show top 3
            similarity_score = float(fact_info.get('similarity', 0))  # Convert to Python float
            st.write(f"**{i}.** {fact_info['fact']}")
            st.progress(similarity_score)
            st.caption(f"Similarity: {similarity_score:.2%}")
            if i < len(claim_result['similar_facts'][:3]):
                st.markdown("---")
      # Confidence score (if available)
    if 'confidence' in claim_result and claim_result['confidence'] > 0:
        st.markdown("**🎯 Confidence Score:**")
        confidence = float(claim_result['confidence'])  # Convert to Python float
        st.progress(confidence)
        st.caption(f"Average similarity to database facts: {confidence:.2%}")
    
    # Feedback section with unique keys
    st.markdown("**📝 Was this helpful?**")
    col1, col2, col3 = st.columns(3)
    
    # Create unique keys to avoid Streamlit conflicts
    claim_key = hashlib.md5(str(claim_result.get('claim', 'unknown')).encode()).hexdigest()[:8]
    
    with col1:
        if st.button("👍 Yes", key=f"helpful_{claim_key}"):
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 No", key=f"not_helpful_{claim_key}"):
            st.info("We'll work on improving our fact-checking!")
    with col3:
        if st.button("🤔 Partially", key=f"partial_{claim_key}"):
            st.info("Thanks! Your feedback helps us improve.")

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Truth Checker</h1>', unsafe_allow_html=True)
    st.markdown("### LLM-Powered Fact Verification System")
    st.markdown("Enter a news statement or social media post to verify its claims against our trusted database.")
    
    # Check if imports were successful
    if not import_success:
        st.error(f"❌ Import Error: {import_error}")
        st.info("Please check that all dependencies are installed properly.")
        return
    
    # Main input area in a single centered container
    st.header("📝 Input Statement")
    input_text = st.text_area(
        "Enter the statement to fact-check:",
        placeholder="Example: The Indian government has announced free electricity to all farmers starting July 2025.",
        height=150
    )
    check_button = st.button("🔍 Check Facts", type="primary", use_container_width=True)
    
    # Process the fact-check
    if check_button and input_text.strip():
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Please provide your OpenAI API key in the .env file.")
            return
            
        with st.spinner("🔍 Analyzing your statement..."):
            try:
                result = fact_check_pipeline(input_text)
                st.session_state.last_result = result
                
                # Display results
                st.markdown("---")
                st.header("📊 Fact-Check Results")
                
                # Handle different result formats
                if 'individual_claims' in result:
                    # Multiple claims
                    overall_verdict = result['overall_verdict']
                    st.markdown(f"### Overall Verdict: {display_verdict_emoji(overall_verdict)} **{overall_verdict}**")
                    
                    st.markdown("### Individual Claims Analysis:")
                    for i, claim_result in enumerate(result['individual_claims'], 1):
                        st.markdown(f"#### Claim {i}: *{claim_result['claim']}*")
                        display_claim_result(claim_result)
                        if i < len(result['individual_claims']):
                            st.markdown("---")
                else:
                    # Single claim
                    if 'claim' in result:
                        st.markdown(f"### Claim: *{result['claim']}*")
                    display_claim_result(result)
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.error("Please check your configuration and try again.")
                # Display the full error for debugging
                with st.expander("🔧 Debug Information"):
                    st.code(str(e))
    
    elif check_button:
        st.warning("⚠️ Please enter a statement to fact-check.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit, spaCy, FAISS, and OpenAI**")

if __name__ == "__main__":
    main()
