import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    batch_size: int = 12
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


@dataclass
class PathConfig:
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"
