import os
import sys
import time
import torch
import argparse
from training import get_loaders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training import load_tokenizer, setup_device
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from training import load_model
from utils.model_utils import *
from torch.utils.data import Dataset, DataLoader, Subset

local_rank, device, use_ddp = setup_device()
@torch.no_grad()
def evaluate(model, test_loader, tokenizer, max_batches=None,device=None):
    start_time = time.time()
    model.eval()
    total_loss = 0
    total_batches = 0

    # total_tokens = 0
    pad_token_id = tokenizer.pad_token_id


    if not dist.is_initialized() or dist.get_rank() == 0:
        if max_batches is None:
            print(f"Evaluating on the full test set...")
        else:
            print(f"Evaluating on up to {max_batches} batches...")

    # Start timing for throughput calculation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # throughput_start_time = time.time()

    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids']
        targets = batch['target_ids']

        # Count tokens processed by the model (input tokens)
        # This represents the actual computational work done in the forward pass
        # if pad_token_id is not None:
        #     # Count non-padding tokens in input (what model actually processes)
        #     non_pad_mask = (input_ids != pad_token_id)
        #     batch_tokens = non_pad_mask.sum().item()
        # else:
        #     # If no pad token, count all input tokens
        #     batch_tokens = input_ids.numel()

        # total_tokens += batch_tokens

        # Compute loss
        _, loss = model(input_ids, targets)
        total_loss += loss.item()
        total_batches += 1

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
             print(f"  Batch {batch_idx + 1}/{len(test_loader)} | test_loss {loss.item():.4f} | test_perplexity {torch.exp(loss).item():.4f}", flush=True)

    # End timing for throughput calculation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    throughput_end_time = time.time()

    # Compute metrics
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")
    print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")
    return avg_loss,avg_perplexity

    # throughput_elapsed = throughput_end_time - throughput_start_time

    # # Calculate throughput
    # tokens_per_second = total_tokens / throughput_elapsed if throughput_elapsed > 0 else 0

    # if not dist.is_initialized() or dist.get_rank() == 0:
    #     print(f"Evaluation completed in {elapsed:.2f} seconds")
    #     print(f"Total Batches Processed: {batch_idx + 1}")
    #     print(f"Total Tokens Processed: {total_tokens:,}")
    #     # print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    #     print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")

    # # return avg_loss, avg_perplexity, tokens_per_second
    # return avg_loss, avg_perplexity

    
def main():
    """
    Main entry point for evaluating the predictive coding transformer model.
    Parses command-line arguments, sets up the model, data, and evaluation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    _ = parser.parse_args()  # Currently unused but kept for future flash attention support

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path,vocab_size).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    _, _, test_loader = get_loaders(distributed=use_ddp)

    # Max batches can be set to limit evaluation, or None for full dataset
    if torch.cuda.is_available():
            torch.cuda.synchronize()
    start_time = time.time()
    # avg_loss, avg_perplexity, throughput = evaluate(model, test_loader, tokenizer, max_batches = None, device = device)
    avg_loss, avg_perplexity = evaluate(model, test_loader, tokenizer, max_batches = None, device = device)
    if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.time() - start_time
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Overall evaluation completed in {elapsed:.2f} seconds")
        # print(f"Final Results - Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}, Throughput: {throughput:.2f} tokens/sec")
       

    if use_ddp and dist.is_initialized():   
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 

