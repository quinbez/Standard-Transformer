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
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bertscore
import time
import argparse
# nltk.download('punkt')
import os
import gc
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def setup_device():
    """Setup device for training with optional DDP support"""
    if "WORLD_SIZE" in os.environ and torch.cuda.is_available():
        # DDP mode
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_ddp = True
        print(f"DDP mode: Using device cuda:{local_rank}")
    elif torch.cuda.is_available():
        # Single GPU mode
        local_rank = 0
        device = torch.device("cuda:0")
        use_ddp = False
        print(f"Single GPU mode: Using device cuda:0")
    else:
        # CPU mode
        local_rank = 0
        device = torch.device("cpu")
        use_ddp = False
        print("CPU mode: Using device cpu")
    return local_rank, device, use_ddp

# Hyperparameters

batch_size = 64
block_size = 256
MAX_LENGTH = 64
learning_rate = 1e-5
n_embd = 64
n_head = 8
n_layer = 4
dropout = 0.1
max_epochs = 20
max_new_tokens = 50
temperature = 1.0
num_workers = 0


# Directory setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "ptbdataset") 
TOKENIZER_DIR = os.path.join(os.path.dirname(DATA_DIR), 'tokenized_ptb')
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# BPE Tokenizer class
class BPETokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]

    def train_and_save(self):
        tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            print(f"Tokenizer already exists at {tokenizer_path}, skipping training.")
            return
        
        train_path = os.path.join(DATA_DIR, "ptb.train.txt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        
        with open(train_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        trainer = trainers.BpeTrainer(
            special_tokens=self.special_tokens,
            vocab_size=4000,
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)

        metadata = {"special_tokens": self.special_tokens}
        metadata_path = os.path.join(TOKENIZER_DIR, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Tokenizer trained and saved to {tokenizer_path}")
        print(f"Metadata saved to {metadata_path}")

    def tokenize_and_save(self, subset_name):
        tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        subset_path = os.path.join(DATA_DIR, f"ptb.{subset_name}.txt")
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f"{subset_name}.txt not found in {DATA_DIR}")
        
        with open(subset_path, "r", encoding="utf-8") as f:
            sep_id = self.tokenizer.token_to_id("[EOS]")
            if sep_id is None:
                raise ValueError("Special token [EOS] not found in tokenizer vocabulary.")
            
            tokenized = [
                self.tokenizer.encode(line.strip()).ids + [sep_id]
                for line in f if line.strip()
            ]

        output_path = os.path.join(TOKENIZER_DIR, f"{subset_name}_ids.pkl")
        if os.path.exists(output_path):
            print(f"Tokenized IDs already exist for {subset_name} at {output_path}, skipping.")
            return 

        with open(output_path, "wb") as f:
            pickle.dump(tokenized, f)

        print(f"Tokenized ptb.{subset_name}.txt and saved IDs to {output_path}")
# Tokenizer initialization will be moved to main() function

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

# def create_data_loaders(use_ddp=False):
#     """Create data loaders with optional DDP support"""
#     # Create datasets
#     train_dataset = PennTreebankDataset("train_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)
#     val_dataset = PennTreebankDataset("valid_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)
#     test_dataset = PennTreebankDataset("test_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)

#     if use_ddp:
#         from torch.utils.data.distributed import DistributedSampler
#         train_sampler = DistributedSampler(train_dataset, shuffle=True)
#         val_sampler = DistributedSampler(val_dataset, shuffle=False)
#         test_sampler = DistributedSampler(test_dataset, shuffle=False)
#         shuffle = False  # Don't shuffle when using DistributedSampler
#     else:
#         train_sampler = None
#         val_sampler = None
#         test_sampler = None
#         shuffle = True

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         sampler=train_sampler,
#         collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
#     )
#     valid_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         sampler=val_sampler,
#         collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         sampler=test_sampler,
#         collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
#     )

#     return train_loader, valid_loader, test_loader
def get_datasets():
    train_dataset = PennTreebankDataset("train_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)
    val_dataset = PennTreebankDataset("valid_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)
    test_dataset = PennTreebankDataset("test_ids.pkl", TOKENIZER_DIR, MAX_LENGTH)

def get_loaders(distributed: bool = False):
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.token_to_id("[PAD]")
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
        pin_memory=True,
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
        pin_memory=True,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=num_workers > 0
    )                     
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=num_workers > 0
    )
    return train_loader, valid_loader, test_loader
# Padding collate function
def pad_collate_fn(batch, pad_token_id=0):
    input_seqs = [item["input_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=pad_token_id)
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_token_id)

    return {"input_ids": input_seqs, "target_ids": target_seqs}

def decode_ids(tokenizer, ids, stop_at_eos = True):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if stop_at_eos and "[EOS]" in text:
        text = text.split("[EOS]")[0].strip()
    return text

# Load tokenizer function
def load_tokenizer():
    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    return Tokenizer.from_file(tokenizer_path)


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
    def __init__(self):
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
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    total_batches = 0

    # Get rank for distributed training
    # rank = dist.get_rank() if dist.is_initialized() else 0

    for batch_idx, batch in enumerate(train_loader):
        xb = batch['input_ids'].to(device)
        yb = batch['target_ids'].to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        # Only print from rank 0 in distributed training
        # if rank == 0 and (batch_idx + 1) % 10 == 0:
        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f}", flush=True)

        # Clean up memory
        cleanup_memory()

    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, avg_perplexity


# def compute_text_metrics(predictions, targets):
#     print("\nComputing BERTScore and BLEU...")
#     P, R, F1 = bertscore(
#         predictions,
#         targets,
#         lang="en",
#         model_type="roberta-base",
#         rescale_with_baseline=True,
#     )
#     print(f"BERTScore (F1): {F1.mean().item():.4f}")

#     smooth_fn = SmoothingFunction().method4
#     tokenized_targets = [[target.split()] for target in targets]
#     tokenized_pred = [pred.split() for pred in predictions]
#     bleu = corpus_bleu(tokenized_targets, tokenized_pred, smoothing_function=smooth_fn)
#     print(f"BLEU Score: {bleu:.4f}")

@torch.no_grad()
def evaluate(model, test_loader, tokenizer, device, max_batches=None, compute_metrics=True):
    start_time = time.time()
    model.eval()
    total_loss = 0
    total_batches = 0
    pad_token_id = tokenizer.token_to_id("[PAD]")
    decoded_targets, decoded_predictions = [], []

    # Get rank for distributed training
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        if max_batches is None:
            print(f"Evaluating on the full test set...")
        else:
            print(f"Evaluating on up to {max_batches} batches...")

    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        targets = batch['target_ids'].to(device)

        # Compute loss
        logits, loss = model(input_ids, targets)
        total_loss += loss.item()
        total_batches += 1

        if compute_metrics:
             preds = torch.argmax(logits, dim=-1)
             mask = targets != pad_token_id
             for i in range(preds.size(0)):
                pred_str = decode_ids(tokenizer, preds[i][mask[i]].cpu().tolist(), stop_at_eos=True)
                tgt_str = decode_ids(tokenizer, targets[i][mask[i]].cpu().tolist(), stop_at_eos=True)
                decoded_predictions.append(pred_str)
                decoded_targets.append(tgt_str)

        # Clean up memory
        cleanup_memory()

    if rank == 0 and compute_metrics and decoded_predictions and decoded_targets:
        compute_text_metrics(decoded_predictions, decoded_targets)

    # Compute average loss and perplexity
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"Evaluation completed in {elapsed:.2f} seconds")
        print(f"Total Batches Processed: {batch_idx + 1}")
        print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")

    return avg_loss, avg_perplexity



def main():
    """Main training function with DDP support"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention (not implemented in this model)')
    args = parser.parse_args()

    # Setup device and DDP
    local_rank, device, use_ddp = setup_device()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Get rank for printing
    # rank = dist.get_rank() if dist.is_initialized() else 0

    # Initialize tokenizer only on rank 0, then broadcast or let others load
    # if rank == 0:
    #     # Initialize and train tokenizer (only on rank 0)
    #     bpe = BPETokenizer()
    #     bpe.train_and_save()

    #     # Tokenize datasets (only on rank 0)
    #     bpe.tokenize_and_save("train")
    #     bpe.tokenize_and_save("valid")
    #     bpe.tokenize_and_save("test")

    # # Synchronize all processes
    # if use_ddp:
    #     dist.barrier()

    # # Load tokenizer (all ranks)
    # tokenizer = load_tokenizer()
    # global vocab_size, pad_token_id, eos_token_id
    # vocab_size = tokenizer.get_vocab_size()
    # pad_token_id = tokenizer.token_to_id("[PAD]")
    # eos_token_id = tokenizer.token_to_id("[EOS]")

   

    # Create model and move to device
    model = LanguageModel().to(device)

    # Wrap model with DDP if needed
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # Create data loaders
    train_loader, valid_loader, test_loader = get_loaders(distributed=use_ddp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_training_time = time.time()
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("Starting training...")
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.5f} M parameters")
       

    for epoch in range(max_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        model.train()
        avg_loss, avg_perplexity = train(model, train_loader, optimizer, epoch, device)

        if rank == 0:
            print(f"Epoch {epoch + 1} completed | avg_train_loss {avg_loss:.4f} | avg_train_perplexity {avg_perplexity:.4f}")

    if rank == 0:
        total_training_time = time.time() - start_training_time
        print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
        print("========== Training completed ==========", flush=True)

        # Save model
        save_path = "checkpoints/gpt_backprop.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)

        # Save the actual model state dict (unwrap DDP if needed)
        model_state = model.module.state_dict() if use_ddp else model.state_dict()
        torch.save({"model_state": model_state}, save_path)
        print("Model saved.")

        # Evaluate with metrics
        # test_loss, test_perplexity = evaluate(model, test_loader, tokenizer, device, max_batches=10, compute_metrics=True)

    # Clean up DDP
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

    # return (model, test_loader, tokenizer, device, pad_token_id, eos_token_id) if rank == 0 else (None, None, None, None, None, None)
    if __name__ == "__main__":
        main()
# def generate(model, input_ids, device, eos_token_id, max_new_tokens=50, temperature=1.0):
#     """Generate text samples from the model"""
#     model.eval()
#     input_tensor = input_ids.unsqueeze(0).to(device)  # Add batch dimension and move to device

#     for _ in range(max_new_tokens):
#         if input_tensor.size(1) > block_size:
#             input_tensor = input_tensor[:, -block_size:]
#         with torch.no_grad():
#             logits, _ = model(input_tensor)
#             logits = logits[:, -1, :] / temperature
#             probs = F.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#             input_tensor = torch.cat((input_tensor, next_token), dim=1)
#         if next_token.item() == eos_token_id:
#             break

#     return input_tensor[0].cpu()  # Move back to CPU for processing


# def run_generation_samples(model, test_loader, tokenizer, device, pad_token_id, eos_token_id):
#     """Run generation samples if we're on rank 0"""
#     for batch_idx, batch in enumerate(test_loader):
#         input_ids = batch["input_ids"]
#         target_ids = batch["target_ids"]
#         break

#     num_samples = 5
#     prompt_len = 4

#     for i in range(num_samples):
#         prompt_ids = input_ids[i][:prompt_len]
#         generated_ids = generate(model, prompt_ids, device, eos_token_id, max_new_tokens=50, temperature=0.7)

#         target_continuation = target_ids[i][prompt_len:]
#         target_continuation = target_continuation[target_continuation != pad_token_id].tolist()

#         generated_continuation = generated_ids[prompt_len:].tolist()

#         # Decode all
#         prompt_str = decode_ids(tokenizer, prompt_ids.tolist())
#         target_str = decode_ids(tokenizer, target_continuation, stop_at_eos=True)
#         predict_str = decode_ids(tokenizer, generated_continuation, stop_at_eos=True)

#         print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
#         print(f"[PROMPT ]: {prompt_str}")
#         print(f"[TARGET ]: {target_str}")
#         print(f"[PREDICT]: {predict_str}")


# if __name__ == "__main__":
#     model, test_loader, tokenizer, device, pad_token_id, eos_token_id = main()

#     # Run generation samples only on rank 0
#     if model is not None:
#         run_generation_samples(model, test_loader, tokenizer, device, pad_token_id, eos_token_id)
