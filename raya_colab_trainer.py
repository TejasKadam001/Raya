"""
Raya - 124M parameter transformer model
Train on Colab with T4 GPU, save checkpoint, and use with the FastAPI backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
import re
from tqdm import tqdm

# Config
class Config:
    vocab_size = 50257
    n_layers = 12
    d_model = 768
    n_heads = 12
    d_ff = 3072
    max_seq_len = 512
    dropout = 0.1
    batch_size = 12
    lr = 3e-4
    weight_decay = 0.1
    epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Attention block with causal masking
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Causal mask for autoregressive generation
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# Transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Main GPT model
class RayaGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie token embedding and output weights
        self.head.weight = self.tok_emb.weight

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return loss, logits

# Aggressive data cleaning to prevent repetitive outputs
def clean_text(text):
    # Remove HTML and URLs
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode()
    
    # Remove lines with high repetition (e.g., "========")
    lines = text.split('.')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty or very short lines
        if len(line) < 10:
            continue
        # Skip lines with repetitive characters
        if len(set(line.replace(' ', ''))) < 5:
            continue
        cleaned_lines.append(line)
    
    return '. '.join(cleaned_lines)

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(1, (len(self.tokens) - 1) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        
        # Pad if needed
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    print("Starting Raya training on Colab...")
    
    os.system('pip install -q tiktoken datasets')
    import tiktoken
    from datasets import load_dataset

    config = Config()
    enc = tiktoken.get_encoding("gpt2")
    
    print("\nLoading and cleaning dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    # Aggressive cleaning
    cleaned_texts = []
    for item in ds['text']:
        cleaned = clean_text(item)
        if len(cleaned) > 50:  # Only keep substantial text
            cleaned_texts.append(cleaned)
    
    full_text = ' '.join(cleaned_texts[:50000])  # Use first 50k samples
    tokens = enc.encode(full_text)
    
    print(f"Cleaned data: {len(tokens):,} tokens")
    
    train_ds = TextDataset(tokens, config.max_seq_len)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    model = RayaGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    print(f"\nTraining on {config.device}...")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for x, y in pbar:
            x, y = x.to(config.device), y.to(config.device)
            
            loss, _ = model(x, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.3f}")
        
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, f'raya_e{epoch+1}.pt')
        
        print(f"Saved: raya_e{epoch+1}.pt")
    
    print("\nTraining finished! Download raya_e5.pt and use with your Raya backend.")
