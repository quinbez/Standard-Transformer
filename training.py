import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import pickle
import json
import nltk
import time
import argparse
import os
import gc
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from utils.model_utils import *
from data_preparation.config import Config
from transformers import GPT2TokenizerFast

def load_model(model_path,vocab_size):
    """
    Load a LanguageModel from a checkpoint file.

    Args:
        model_path (str): Path to the saved model checkpoint.
        vocab_size (int): Vocabulary size for the model.
    Returns:
        LanguageModel: The loaded model with trained weights.
    """
    model = LanguageModel(vocab_size)
    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # Assume the dict itself is the state dict
            model.load_state_dict(checkpoint, strict=False)
    else:
        # Assume it's directly the state dict
        model.load_state_dict(checkpoint, strict=False)

    print(f"Successfully loaded model from {model_path}")
    return model


# Hyperparameters
batch_size = 64
block_size = 256
MAX_LENGTH = 64
learning_rate = 1e-5
n_embd = 64
n_head = 2
n_layer = 2
dropout = 0.1
max_epochs = 2
max_new_tokens = 50
temperature = 1.0
num_workers = 0

# Directory setup

# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# DATA_DIR = os.path.join(BASE_DIR, "ptbdataset") 
# TOKENIZER_DIR = os.path.join(os.path.dirname(DATA_DIR), 'tokenized_ptb')
# os.makedirs(TOKENIZER_DIR, exist_ok=True)

# # BPE Tokenizer class
# class BPETokenizer:
#     def __init__(self):
#         self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
#         self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#         self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]

#     def train_and_save(self):
#         tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#         if os.path.exists(tokenizer_path):
#             print(f"Tokenizer already exists at {tokenizer_path}, skipping training.")
#             return
        
#         train_path = os.path.join(DATA_DIR, "ptb.train.txt")
#         if not os.path.exists(train_path):
#             raise FileNotFoundError(f"Train file not found: {train_path}")
        
#         with open(train_path, "r", encoding="utf-8") as f:
#             sentences = [line.strip() for line in f if line.strip()]

#         trainer = trainers.BpeTrainer(
#             special_tokens=self.special_tokens,
#             vocab_size=4000,
#             min_frequency=2
#         )
#         self.tokenizer.train_from_iterator(sentences, trainer=trainer)
#         tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#         self.tokenizer.save(tokenizer_path)

#         metadata = {"special_tokens": self.special_tokens}
#         metadata_path = os.path.join(TOKENIZER_DIR, "metadata.json")
#         with open(metadata_path, "w", encoding="utf-8") as f:
#             json.dump(metadata, f, indent=4)

#         print(f"Tokenizer trained and saved to {tokenizer_path}")
#         print(f"Metadata saved to {metadata_path}")

#     def tokenize_and_save(self, subset_name):
#         tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#         if not os.path.exists(tokenizer_path):
#             raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
#         subset_path = os.path.join(DATA_DIR, f"ptb.{subset_name}.txt")
#         if not os.path.exists(subset_path):
#             raise FileNotFoundError(f"{subset_name}.txt not found in {DATA_DIR}")
        
#         with open(subset_path, "r", encoding="utf-8") as f:
#             sep_id = self.tokenizer.token_to_id("[EOS]")
#             if sep_id is None:
#                 raise ValueError("Special token [EOS] not found in tokenizer vocabulary.")
            
#             tokenized = [
#                 self.tokenizer.encode(line.strip()).ids + [sep_id]
#                 for line in f if line.strip()
#             ]

#         output_path = os.path.join(TOKENIZER_DIR, f"{subset_name}_ids.pkl")
#         if os.path.exists(output_path):
#             print(f"Tokenized IDs already exist for {subset_name} at {output_path}, skipping.")
#             return 

#         with open(output_path, "wb") as f:
#             pickle.dump(tokenized, f)

#         print(f"Tokenized ptb.{subset_name}.txt and saved IDs to {output_path}")
# # Tokenizer initialization will be moved to main() function

# Penn Treebank Dataset class
class PennTreebankDataset(Dataset):
    def __init__(self, tokenized_file, tokenizer_dir, block_size):
        self.tokenizer_dir = tokenizer_dir
        self.block_size = block_size

        tokenized_file_path = os.path.join(self.tokenizer_dir, tokenized_file)
        if not os.path.exists(tokenized_file_path):
            raise FileNotFoundError(f"Tokenized file not found: {tokenized_file_path}")

        with open(tokenized_file_path, 'rb') as f:
            self.sequences = pickle.load(f)

        self.sequences = [seq for seq in self.sequences if len(seq) > 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1][:self.block_size], dtype=torch.long)
        target_ids = torch.tensor(seq[1:][:self.block_size], dtype=torch.long)
        
        return {"input_ids": input_ids, "target_ids": target_ids}
# Load tokenizer function
# def load_tokenizer():
#     tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
#     return Tokenizer.from_file(tokenizer_path)
def load_tokenizer():
    tokenizer_path = os.path.join(Config.TOKENIZER_DIR, f"gpt2_tokenizer_{Config.DATASET_NAME}.json")
    tokenizer= GPT2TokenizerFast.from_pretrained(tokenizer_path)
    special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(special_tokens)
    
    Config.VOCAB_SIZE = len(tokenizer) 
    Config.PAD_ID = tokenizer.pad_token_id
    Config.EOS_ID = tokenizer.eos_token_id
    return tokenizer
def get_datasets():
    train_dataset = PennTreebankDataset("ptb_train_ids.pkl", Config.TOKENIZER_DIR, MAX_LENGTH)
    valid_dataset = PennTreebankDataset("ptb_valid_ids.pkl", Config.TOKENIZER_DIR, MAX_LENGTH)
    test_dataset = PennTreebankDataset("ptb_test_ids.pkl", Config.TOKENIZER_DIR, MAX_LENGTH)
    return train_dataset, valid_dataset, test_dataset

def get_loaders(distributed: bool = False):
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.pad_token_id
    train_dataset, valid_dataset, test_dataset = get_datasets()

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = valid_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=num_workers > 0
    )                     
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=num_workers > 0
    )
    return train_loader, valid_loader, test_loader

# Global variables will be set in main() function
vocab_size = None
pad_token_id = None
eos_token_id = None

# Model architecture
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

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
            nn.Dropout(dropout),
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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


# Training function
def train(model, train_loader, optimizer, epoch, device, tokenizer=None):
    model.train()
    total_loss = 0
    total_batches = 0
    total_tokens = 0
    pad_token_id = tokenizer.pad_token_id if tokenizer else None

    # Start timing for throughput calculation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch['input_ids'].to(device)
        yb = batch['target_ids'].to(device)

        # Count tokens processed by the model (input tokens)
        if pad_token_id is not None:
            # Count non-padding tokens in input (what model actually processes)
            non_pad_mask = (xb != pad_token_id)
            batch_tokens = non_pad_mask.sum().item()
        else:
            # If no pad token, count all input tokens
            batch_tokens = xb.numel()

        total_tokens += batch_tokens

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            # Calculate current throughput
            elapsed_so_far = time.time() - start_time
            current_throughput = total_tokens / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f} | throughput {current_throughput:.0f} tokens/sec", flush=True)

        # Clean up memory
        cleanup_memory()

    # End timing for throughput calculation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    # Calculate final metrics
    elapsed_time = end_time - start_time
    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

    return avg_loss, avg_perplexity, tokens_per_second, total_tokens, elapsed_time


def main():
    """Main training function with DDP support"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention (not implemented in this model)')
    _ = parser.parse_args()  # Currently unused but kept for future flash attention support

    # Setup device and DDP
    local_rank, device, use_ddp = setup_device()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    tokenizer = load_tokenizer()
    # vocab_size = tokenizer.get_vocab_size()
    vocab_size = len(tokenizer)
    # Create model and move to device
    model = LanguageModel(vocab_size=vocab_size).to(device)

    # Wrap model with DDP if needed
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # Create data loaders
    train_loader, _, _ = get_loaders(distributed=use_ddp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
   
    start_training_time = time.time()
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("Starting training...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.5f} M parameters")
       

    for epoch in range(max_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        model.train()
        avg_loss, avg_perplexity, throughput, total_tokens, epoch_time = train(model, train_loader, optimizer, epoch, device, tokenizer)


        if rank == 0:
            print(f"Epoch {epoch + 1} completed | avg_train_loss {avg_loss:.4f} | avg_train_perplexity {avg_perplexity:.4f} | throughput {throughput:.0f} tokens/sec | tokens {total_tokens:,} | time {epoch_time:.1f}s")

    if rank == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_training_time = time.time() - start_training_time
        print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
        print("========== Training completed ==========", flush=True)

        # Save model
        save_path = "checkpoints/final_model.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)

        # Save the actual model state dict (unwrap DDP if needed)
        model_state = model.module.state_dict() if use_ddp else model.state_dict()
        torch.save({"model_state": model_state}, save_path)
        print("Model saved.")

    # Clean up DDP
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
