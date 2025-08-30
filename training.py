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
def train(model, train_loader, optimizer, epoch, config, device):
    model.train()
    total_loss = 0
    total_batches = 0
    # total_tokens = 0

    # Start timing for throughput calculation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch['input_ids'].to(device)
        yb = batch['target_ids'].to(device)

        # # Count tokens processed by the model (input tokens)
        # if config.pad_token_id is not None:
        #     # Count non-padding tokens in input (what model actually processes)
        #     non_pad_mask = (xb != config.pad_token_id)
        #     batch_tokens = non_pad_mask.sum().item()
        # else:
        #     # If no pad token, count all input tokens
        #     batch_tokens = xb.numel()

        # total_tokens += batch_tokens

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        perplexity = torch.exp(loss).item()

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
        #     # Calculate current throughput
        #     elapsed_so_far = time.time() - start_time
        #     current_throughput = total_tokens / elapsed_so_far if elapsed_so_far > 0 else 0
        #     print(f"  Batch {batch_idx + 1}/{len(train_loader)} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f} | throughput {current_throughput:.0f} tokens/sec", flush=True)
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | train_loss {loss:.4f} | train_perplexity {perplexity:.4f}", flush=True)

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
    # tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0


    # return avg_loss, avg_perplexity, tokens_per_second, total_tokens, elapsed_time
    return avg_loss, avg_perplexity


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
    vocab_size = len(tokenizer)

    # Hyperparameters
    config = GPTConfig(
        vocab_size = vocab_size,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        block_size = 176, 
        learning_rate = 2.5e-05,
        n_embd=112,
        n_head = 7,
        n_layer = 3,
        dropout= 0.149,
        max_epochs = 10,
        max_new_tokens = 50,
        temperature = 1.0
    )

    # Create model and move to device
    model = LanguageModel(config).to(device)

    # Wrap model with DDP if needed
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create data loaders
    train_loader, _, _ = get_loaders(distributed=use_ddp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
   
    start_training_time = time.time()
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("========== Starting training ==========")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.5f} M parameters")
       
    for epoch in range(config.max_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.max_epochs}")
        
        model.train()

        avg_loss, avg_perplexity = train(model, train_loader, optimizer, epoch, config, device)
   
            # avg_loss, avg_perplexity, throughput, total_tokens, epoch_time = train(model, train_loader, optimizer, epoch, device, tokenizer)


        if rank == 0:
    #         print(f"Epoch {epoch + 1} completed | avg_train_loss {avg_loss:.4f} | avg_train_perplexity {avg_perplexity:.4f} | throughput {throughput:.0f} tokens/sec | tokens {total_tokens:,} | time {epoch_time:.1f}s")
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
