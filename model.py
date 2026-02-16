import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._report_parameters()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _report_parameters(self):
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params / 1e6:.2f}M parameters")
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.emb_dropout(token_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return (loss, logits) if targets is not None else logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits = self(input_ids_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
