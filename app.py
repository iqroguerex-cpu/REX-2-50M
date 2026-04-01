import streamlit as st
import torch
from transformers import AutoTokenizer
from model import IQ_Model # Import the same brain

st.set_page_config(page_title="IQROGUEREX StoryGen", page_icon="🤖")
st.title("🚀 IQROGUEREX Story Engine")

@st.cache_resource
def load_all():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = IQ_Model(tokenizer.vocab_size).to(device)
    # Ensure this filename matches your uploaded weights
    model.load_state_dict(torch.load('iq_model_stream_final.pth', map_location=device))
    return model, tokenizer, device

model, tokenizer, device = load_all()

prompt = st.text_input("Enter a prompt:", "Timmy went to the")
temp = st.slider("Creativity", 0.1, 1.0, 0.7)

if st.button("Generate"):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=100, temperature=temp, device=device)
    st.write(tokenizer.decode(output[0], skip_special_tokens=True))
