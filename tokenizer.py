import tiktoken
from typing import List

class SimpleTokenizer:
    """Wrapper around tiktoken for GPT-2 tokenization"""
    
    def __init__(self, vocab_size: int = 50257):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)
