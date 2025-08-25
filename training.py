import os
import time
import torch
import argparse
from utils.model_utils import *
import torch.distributed as dist
from model_architecture.config import GPTConfig
from model_architecture.model import LanguageModel
from torch.nn.parallel import DistributedDataParallel as DDP

# Training function
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    total_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        xb = batch['input_ids'].to(device)
        yb = batch['target_ids'].to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f}", flush=True)

        # Clean up memory
        cleanup_memory()

    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
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
    vocab_size = len(tokenizer)
    
    # Create model and move to device
    model = LanguageModel(vocab_size=vocab_size).to(device)

    # Wrap model with DDP if needed
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = get_loaders(distributed=use_ddp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=GPTConfig.learning_rate)
   
    start_training_time = time.time()
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("========== Starting training ==========")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.5f} M parameters")
       
    for epoch in range(GPTConfig.max_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{GPTConfig.max_epochs}")
        
        model.train()
        avg_loss, avg_perplexity = train(model, train_loader, optimizer, epoch, device)
        
        if rank == 0:
            print(f"Epoch {epoch + 1} completed | avg_train_loss {avg_loss:.4f} | avg_train_perplexity {avg_perplexity:.4f}")

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
