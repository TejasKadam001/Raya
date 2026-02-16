import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Raya Architecture (124M Params)
class RayaModel(nn.Module):
    def __init__(self, vocab_size=50257, n_layers=12, d_model=768, n_heads=12, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Use ModuleList with key name 'blocks' to match checkpoint
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        # Embedding
        x = self.tok_emb(x) + self.pos_emb(pos)
        
        # Match chat_gui.py: no mask (bidirectional)
        for block in self.blocks:
            x = block(x)
        
        logits = self.head(self.ln(x))
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == 50256: # EOS token
                break
        return input_ids


class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 100

class ChatResponse(BaseModel):
    response: str

# Global model state
model = None
enc = tiktoken.get_encoding("gpt2")
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def load_model():
    global model
    model_paths = ['code_gpt_e3.pt', 'model_e3.pt', 'checkpoints/best_model.pt', 'code_model_epoch5.pt']
    found_path = None
    for p in model_paths:
        if os.path.exists(p):
            found_path = p
            break
    
    if not found_path:
        print("⚠️ Model file not found.")
        return False

    print(f"Loading {found_path}...")
    checkpoint = torch.load(found_path, map_location=device)
    model = RayaModel().to(device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint['model_state_dict']
    
    # Use non-strict loading to ignore the 'mask' buffer and handle architecture differences
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Raya is Ready (on {device.upper()})")
    return True

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    load_model()
    yield
    # Clean up on shutdown if needed

app = FastAPI(title="Raya AI API", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tokens = enc.encode(request.message)
    x = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            x, 
            max_new_tokens=request.max_tokens, 
            temperature=request.temperature, 
            top_k=request.top_k
        )
        response_text = enc.decode(output_ids[0].tolist())
        
        # Remove prompt from output
        if response_text.startswith(request.message):
            response_text = response_text[len(request.message):].strip()

    return ChatResponse(response=response_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
