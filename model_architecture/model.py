import torch
import torch.nn as nn
from torch.nn import functional as F
from model_architecture.config import GPTConfig

# Model architecture
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(GPTConfig.n_embd, head_size, bias=False)
        self.query = nn.Linear(GPTConfig.n_embd, head_size, bias=False)
        self.value = nn.Linear(GPTConfig.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(GPTConfig.block_size, GPTConfig.block_size)))
        self.dropout = nn.Dropout(GPTConfig.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(GPTConfig.n_embd, GPTConfig.n_embd)
        self.dropout = nn.Dropout(GPTConfig.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(GPTConfig.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, GPTConfig.n_embd)
        self.position_embedding_table = nn.Embedding(GPTConfig.block_size, GPTConfig.n_embd)
        self.blocks = nn.Sequential(*[Block(GPTConfig.n_embd, n_head=GPTConfig.n_head) for _ in range(GPTConfig.n_layer)])
        self.ln_f = nn.LayerNorm(GPTConfig.n_embd)
        self.lm_head = nn.Linear(GPTConfig.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss
