import os
import json
import logging

import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
load_dotenv()

# ----------------------
# Configuration & Setup
# ----------------------

# Streamlit page config
st.set_page_config(
    page_title="Mental-Health Q&A Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging to stdout for observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Load environment variables
LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "https://your-claude-proxy.example.com/generate")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in your environment

# ----------------------
# Data & Index Loading
# ----------------------

@st.cache_resource
def load_data_and_index(csv_path: str = "Dataset.csv"):
    """Load dataset and create dual embeddings for questions and answers."""
    try:
        # Load and clean dataset
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["Context", "Response"])
        df = df.drop_duplicates(subset=["Context", "Response"])
        if "Context" not in df.columns or "Response" not in df.columns:
            st.error("❌ Dataset.csv must have 'Context' and 'Response' columns!")
            return None, None, None, None, None
        if len(df) == 0:
            st.error("❌ No valid data found in Dataset.csv!")
            return None, None, None, None, None

        # Initialize embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Compute embeddings for all questions and answers
        questions = df["Context"].astype(str).tolist()
        answers = df["Response"].astype(str).tolist()
        q_embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
        a_embeddings = model.encode(answers, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(q_embeddings)
        faiss.normalize_L2(a_embeddings)

        # Build FAISS indices
        dim = q_embeddings.shape[1]
        q_index = faiss.IndexFlatIP(dim)
        a_index = faiss.IndexFlatIP(dim)
        q_index.add(q_embeddings)
        a_index.add(a_embeddings)

        # Remove or comment out the st.success message
        # st.success(f"✅ Successfully loaded {len(df)} Q&A pairs and built dual search indices!")
        logging.info(f"Loaded {len(df)} Q&A pairs and built dual FAISS indices (dim={dim}).")
        return df, model, q_index, a_index, (q_embeddings, a_embeddings)
    except FileNotFoundError:
        st.error("❌ Dataset.csv not found! Please add your CSV file with 'Context' and 'Response' columns.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        logging.error(f"Error loading dataset: {e}")
        return None, None, None, None, None

# Load data and create indices
df, embed_model, q_faiss_index, a_faiss_index, (q_emb_matrix, a_emb_matrix) = load_data_and_index()

# ----------------------
# Helper Functions
# ----------------------

def moderate_text(text: str) -> bool:
    """
    Placeholder for safety moderation via Perspective API or other.
    Return False if text should be blocked/redacted.
    """
    # Implement real call if PERSPECTIVE_API_KEY is set
    return True

def stream_openai_response(prompt: str, model: str = "gpt-3.5-turbo"):
    """
    Stream tokens from OpenAI's GPT model using the new openai>=1.0.0 API.
    Yields text chunks.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an empathetic mental-health peer supporter — not a licensed therapist."},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        max_tokens=500,
        temperature=0.7,
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def search_similar_qa(query: str, k: int = 5):
    """Dual-embedding similarity: average query-to-question and query-to-answer similarity, return top k unique Q/A pairs."""
    try:
        # Embed the query
        query_emb = embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        # Compute similarity to all questions and all answers
        D_q = np.dot(q_emb_matrix, query_emb.T).flatten()  # (N,)
        D_a = np.dot(a_emb_matrix, query_emb.T).flatten()  # (N,)
        combined_score = (D_q + D_a) / 2.0

        # Get top 20 indices by combined score
        top_indices = np.argsort(combined_score)[::-1][:20]

        # Collect top k unique Q/A pairs
        seen = set()
        results = []
        for idx in top_indices:
            context = str(df.iloc[idx]["Context"])
            response = str(df.iloc[idx]["Response"])
            key = (context, response)
            if key not in seen:
                seen.add(key)
                results.append({
                    "Context": context,
                    "Response": response,
                    "Score": float(combined_score[idx])
                })
            if len(results) >= k:
                break
        return results
    except Exception as e:
        st.error(f"❌ Error searching: {str(e)}")
        return []

# ----------------------
# Streamlit UI
# ----------------------

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved" not in st.session_state:
    st.session_state.retrieved = []

# Check if data is loaded
if df is None:
    st.error("❌ Cannot start chat without dataset. Please ensure Dataset.csv exists with 'Context' and 'Response' columns.")
    st.stop()

# Two-column layout: chat and Q&A sidebar
col_chat, col_sidebar = st.columns([3, 1], gap="medium")

# Minimal sidebar with dataset info and similar Q&A
with st.sidebar:
    st.markdown("<h4 style='margin-bottom:0.5em;'>📊 Dataset Info</h4>", unsafe_allow_html=True)
    st.write(f"Total Q&A pairs: {len(df)}")
    if len(df) > 0:
        st.write("Sample question:")
        st.write(f"• {df['Context'].iloc[0][:60]}...")
    st.markdown("---")
    st.markdown("<h4 style='margin-bottom:0.5em;'>🔍 Similar Q&A</h4>", unsafe_allow_html=True)
    if st.session_state.retrieved:
        for i, qa in enumerate(st.session_state.retrieved):
            with st.expander(f"Q: {qa['Context'][:60]}... (Score: {qa['Score']:.3f})"):
                st.write(f"A: {qa['Response'][:200]}...")
    else:
        st.info("Ask a question to see similar Q&A pairs here!", icon="💡")

# Place the Compass title as a centered header above the chat window
st.markdown("""
<div style='text-align:center; margin-bottom: 1.5em;'>
  <span style='font-size:2.5em; font-weight:700; color:#2d2d38;'>🧠 Compass <span style='font-size:0.6em; font-weight:400;'>(Guiding to clarity)</span></span>
</div>
""", unsafe_allow_html=True)

# Main area: clean, minimal
# Remove the old main area title
# st.markdown("<h2 style='margin-bottom:0.5em;'>🧠 Mental Health Q&A Chat</h2>", unsafe_allow_html=True)

with col_chat:
    # Only the clear chat button, chat history, and input
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            styled_advice = f"""
            <div style='background-color: #f0f4fa; border-radius: 12px; padding: 24px 18px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); font-size: 1.15em; color: #222; font-family: "Segoe UI", "Arial", sans-serif;'>
            {msg["content"]}
            </div>
            """
            st.markdown(styled_advice, unsafe_allow_html=True)
    user_input = st.chat_input("How are you feeling today?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        logging.info(f"User: {user_input}")
        results = search_similar_qa(user_input)
        st.session_state.retrieved = results
        logging.info(f"Retrieved top-{len(results)} Q&A pairs.")
        st.rerun()

    # Place Generate Advice and Clear Chat buttons side by side
    col_advice, col_clear = st.columns([3, 1])
    with col_advice:
        generate_clicked = st.button("🤖 Generate Advice", type="primary")
    with col_clear:
        clear_clicked = st.button("🧹 Clear Chat", key="clear_chat_btn")

    if clear_clicked:
        st.session_state.messages = []
        st.session_state.retrieved = []
        st.rerun()

    if generate_clicked:
        if st.session_state.messages:
            user_query = st.session_state.messages[-1]["content"]
            # Always use the top 5 retrieved Q&A pairs
            chosen = st.session_state.retrieved[:5]
            context_text = "\n\n".join([
                f"Q: {qa['Context']}\nA: {qa['Response']}"
                for qa in chosen
            ])
            full_prompt = f"""
You are an empathetic mental health peer supporter — not a licensed therapist.

### Relevant Q&A Context:
{context_text}

### User's Question:
{user_query}

### Task:
Using the emotional tone and practical advice from the above 5 Q&A pairs, compose a single, concise, emotionally aware, and actionable paragraph (no more than 500 characters) in response to the user's question. Be empathetic, use emotional language, and encourage positive action. If the user shows signs of crisis or suicidal ideation, gently encourage professional help.

### Response:
"""
            streaming_text = ""
            with st.spinner("Generating personalized advice..."):
                for chunk in stream_openai_response(full_prompt):
                    if moderate_text(chunk):
                        streaming_text += chunk
            # Remove the last message if it's an assistant message
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages.pop()
            st.session_state.messages.append({"role": "assistant", "content": streaming_text})
            logging.info("Assistant response completed and appended to chat state.")
            st.rerun() 