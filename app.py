import streamlit as st
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from model import IQ_Model 

# --- PAGE SETUP ---
st.set_page_config(page_title="IQROGUEREX REX-2", page_icon="🦖")

# Custom CSS for the "Final Polish" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ENGINE ---
@st.cache_resource
def load_rex_engine():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Hugging Face Config
        REPO_ID = "IqRogueRex/REX-2-50M"
        FILENAME = "iq_model_stream_final.pth"
        
        with st.spinner("Synchronizing REX-2 Weights..."):
        weights_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=FILENAME, 
        token=False,            # <--- Forces NO token usage
        local_files_only=False, # <--- Forces a check against the web
        force_download=True     # <--- Bypasses any "401" cached files
         )
        
        # Initialize and Load
        model = IQ_Model(tokenizer.vocab_size).to(device)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Engine Startup Failed: {e}")
        return None, None, None

# --- RUN LOAD ---
model, tokenizer, device = load_rex_engine()

# --- UI INTERFACE ---
st.title("🦖 REX-2 Story Engine")
st.write("A custom 50M Parameter Transformer by **IQROGUEREX**.")

# Sidebar Controls
st.sidebar.header("Generation Settings")
temp = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.7)
max_len = st.sidebar.slider("Max Story Length", 50, 300, 100)

# Input
user_prompt = st.text_area("Enter your story starter:", "One day, a tiny robot named REX")

# Action
if st.button("Generate with REX-2"):
    if model is not None and tokenizer is not None:
        with st.spinner("REX-2 is generating..."):
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=max_len, 
                temperature=temp, 
                device=device
            )
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            st.subheader("Result:")
            st.info(result)
    else:
        st.error("Model engine is currently offline. Check your Hugging Face Repository.")

st.divider()
st.caption("Built by Chinmay V Chatradamath | IQROGUEREX 2026")
