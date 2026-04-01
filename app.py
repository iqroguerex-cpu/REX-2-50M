import streamlit as st
import torch
import os
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from model import IQ_Model  # Importing your uploaded architecture

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IQROGUEREX | REX-2-50M",
    page_icon="🦖",
    layout="centered"
)

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_name_with_html=True)

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_iq_engine():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 1. Download weights from your HF Repo
    REPO_ID = "IqRogueRex/REX-2-50M"
    FILENAME = "iq_model_stream_final.pth"
    
    try:
        with st.spinner("Synchronizing REX-2 Weights from IQROGUEREX Hub..."):
            weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        # 2. Initialize Architecture
        model = IQ_Model(tokenizer.vocab_size).to(device)
        
        # 3. Load Weights
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face: {e}")
        return None, None, None

model, tokenizer, device = load_iq_engine()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("REX-2 Engine Settings")
temperature = st.sidebar.slider("Creativity (Temp)", 0.1, 1.5, 0.7)
max_tokens = st.sidebar.slider("Story Length", 20, 200, 100)

# --- MAIN UI ---
st.title("🦖 REX-2 Story Engine")
st.caption("A 50M Parameter Transformer by IQROGUEREX")

user_input = st.text_area("Enter a story prompt:", "Once upon a time, REX the robot found a", height=100)

if st.button("Generate with REX-2"):
    if model and tokenizer:
        with st.spinner("REX-2 is dreaming..."):
            # Encode
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
            
            # Generate
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                device=device
            )
            
            # Decode
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            st.subheader("REX-2 Generated Output:")
            st.info(response)
    else:
        st.error("Model engine is offline. Check Hugging Face connection.")

st.divider()
st.markdown("© 2026 **IQROGUEREX** | Lead Engineer: Chinmay V Chatradamath")
