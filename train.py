import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import math
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import tiktoken

from model import GPT
from config import ModelConfig, TrainingConfig, PathConfig
from dataset import create_dataloaders


class Trainer:
    """Language model trainer with stability and precision optimizations"""
    
    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        path_config: PathConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.config = training_config
        self.paths = path_config
        self.device = training_config.device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=training_config.max_epochs * len(train_loader)
        )
        
        self.best_val_loss = float('inf')
        self.step = 0
        self.epoch = 0

    def save_checkpoint(self, val_loss, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config,
            'epoch': self.epoch,
            'val_loss': val_loss
        }
        os.makedirs(self.paths.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.paths.checkpoint_dir, filename))

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            loss, _ = self.model(x, y)
            loss.backward()
            
            # Gradient clipping to prevent noise
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            loss, _ = self.model(x, y)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        print("\n" + "=" * 60)
        print("🚀 Starting High-Accuracy Training")
        print("=" * 60)
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(val_loss, "best_model.pt")
                print(f"✓ New best model saved")
            
        print("\n" + "=" * 60)
        print("🎉 Training Complete!")
        print("=" * 60)


def main():
    mc = ModelConfig()
    tc = TrainingConfig()
    pc = PathConfig()
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Ensure data exists or provide instructions
    train_path = os.path.join(pc.data_dir, "train.txt")
    val_path = os.path.join(pc.data_dir, "val.txt")
    
    if not os.path.exists(train_path):
        print(f"❌ Error: {train_path} not found. Please place your training data in the data/ folder.")
        return

    train_loader, val_loader = create_dataloaders(
        train_path, val_path, enc, tc.batch_size, mc.max_seq_len
    )
    
    model = GPT(mc).to(tc.device)
    trainer = Trainer(model, train_loader, val_loader, mc, tc, pc)
    trainer.fit()


if __name__ == "__main__":
    main()
