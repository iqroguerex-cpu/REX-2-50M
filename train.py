import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from model import IQ_Model  # Import our architecture

# Settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Stream Data
remote_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

model = IQ_Model(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
scaler = torch.amp.GradScaler('cuda')

print("Starting IQROGUEREX Training Stream...")
model.train()

# Training loop logic
for step in range(3000):
    # Fetch data, forward, backward, step (Simplified for repo display)
    # ... (Insert your streaming batch logic here) ...
    if step % 500 == 0:
        torch.save(model.state_dict(), f'checkpoint_{step}.pth')

torch.save(model.state_dict(), 'iq_model_final.pth')
