import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import Any

def clean_text(text: str) -> str:
    """Remove noise and repetitive patterns from training data"""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-ASCII
    text = text.encode("ascii", "ignore").decode()
    
    # Filter out repetitive lines (like ===== or ----)
    lines = text.split('.')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        # Skip lines with lots of repeated chars
        if len(set(line.replace(' ', ''))) < 5:
            continue
        clean_lines.append(line)
    
    return '. '.join(clean_lines)


class TextDataset(Dataset):
    """Simple dataset for text files with sliding windows"""
    
    def __init__(self, text_path: str, tokenizer: Any, max_seq_len: int = 512, sanitize: bool = True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if sanitize:
            text = clean_text(text)
            
        self.tokens = tokenizer.encode(text)
        self.num_samples = max(1, (len(self.tokens) - 1) // max_seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len
        chunk = self.tokens[start_idx:end_idx + 1]
        
        # Pad if necessary
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [0] * (self.max_seq_len + 1 - len(chunk))
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_dataloaders(train_path, val_path, tokenizer, batch_size, max_seq_len):
    """Create train and validation dataloaders"""
    train_ds = TextDataset(train_path, tokenizer, max_seq_len)
    val_ds = TextDataset(val_path, tokenizer, max_seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
