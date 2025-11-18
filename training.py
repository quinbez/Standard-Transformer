import os
import time
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.model_utils import cleanup_memory, setup_device
from model_architecture.config import GPTConfig
from model_architecture.model import LanguageModel
from data_preparation.dataloader import get_loaders
from data_preparation.config import vocab_size

# Training Step
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss, total_batches = 0, 0

    # Sync GPU for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch['input_ids'].to(device)
        yb = batch['target_ids'].to(device)

        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        perplexity = torch.exp(loss).item()

        # Print status from rank 0 every 10 batches
        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                f"train_loss {loss:.4f} | train_perplexity {perplexity:.4f}")

        cleanup_memory()

    # Sync before returning stats
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, avg_perplexity

# Main Training Loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', 
                        help='Enable FlashAttention (not implemented in this model)')
    
    parser.parse_args()  
    
    # Setup device and DDP
    local_rank, device, use_ddp = setup_device()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    # Model configuration
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = 256, 
        learning_rate = 3e-4,
        n_embd=256,
        n_head = 8,
        n_layer = 6,
        dropout= 0.1,
        max_epochs = 1,
        max_new_tokens = 50,
        temperature = 0.8
    )

    # Model
    model = LanguageModel(config).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Data
    train_loader, _, _ = get_loaders(distributed=use_ddp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
       
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Print model info from rank 0
    if rank == 0:
        print("========== Starting training ==========")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{num_params:.5f}M parameters")
      
    start_time = time.time()
    
    # Training epochs 
    for epoch in range(config.max_epochs):

        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.max_epochs}")
        
        avg_loss, avg_perplexity = train(model, train_loader, optimizer, device)

        if rank == 0:
            print(
                f"Epoch {epoch + 1} completed | "
                f"avg_train_loss {avg_loss:.4f} | "
                f"avg_train_perplexity {avg_perplexity:.4f}"
            )
    
    # Final summary
    if rank == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start_time
        print(f"Total Training Time: {total_time:.2f} seconds")
        print("========== Training completed ==========")

        # Save checkpoint
        save_path = "checkpoints/final_model.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_state = model.module.state_dict() if use_ddp else model.state_dict()
        torch.save({"model_state": model_state}, save_path)
        
        print("Model saved.")

    # Cleanup
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
