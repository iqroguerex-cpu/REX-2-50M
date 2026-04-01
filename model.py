import torch
import torch.nn as nn
from torch.nn import functional as F

# Configuration for the 50M Parameter Model
BLOCK_SIZE = 256
N_EMBD = 512
N_HEAD = 8
N_LAYER = 6

class IQ_Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None, device='cpu'):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, device='cpu'):
        self.eval() 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond, device=device)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
