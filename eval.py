import time
import math
import torch
from main import get_loaders
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import load_tokenizer,load_model,setup_device, cleanup_memory,decode_ids,compute_text_metrics
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bertscore
import argparse


local_rank, device, use_ddp = setup_device()
@torch.no_grad()
def evaluate(model, test_loader, tokenizer, max_batches=None,device=None):
    start_time = time.time()
    model.eval()
    total_loss = 0
    total_batches = 0
    pad_token_id = tokenizer.token_to_id("[PAD]")
    decoded_targets, decoded_predictions = [], []
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        if max_batches is None:
            print(f"Evaluating on the full test set...")
        else:
            print(f"Evaluating on up to {max_batches} batches...")

    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids']
        targets = batch['target_ids']

        # Compute loss
        logits, loss = model(input_ids, targets)
        total_loss += loss.item()
        total_batches += 1
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
             print(f"  Batch {batch_idx + 1}/{len(test_loader)} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f}", flush=True)


    #     if compute_metrics:
    #          preds = torch.argmax(logits, dim=-1)
    #          mask = targets != pad_token_id
    #          for i in range(preds.size(0)):
    #             pred_str = decode_ids(tokenizer, preds[i][mask[i]].tolist(), stop_at_eos=True)
    #             tgt_str = decode_ids(tokenizer, targets[i][mask[i]].tolist(), stop_at_eos=True)
    #             decoded_predictions.append(pred_str)
    #             decoded_targets.append(tgt_str)

           
    # if compute_metrics and decoded_predictions and decoded_targets:
    #     compute_text_metrics(decoded_predictions, decoded_targets)
           

    # Compute average loss and perplexity
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")
    print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")
    return avg_loss,avg_perplexity

    
def main():
    """
    Main entry point for evaluating the predictive coding transformer model.
    Parses command-line arguments, sets up the model, data, and evaluation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path,vocab_size).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    _, _, test_loader = get_loaders(distributed=use_ddp)

    # Max batches can be set to limit evaluation, or None for full dataset
    if torch.cuda.is_available():
            torch.cuda.synchronize()
    start_time = time.time()
    evaluate(model, test_loader, tokenizer, max_batches = None, device = device)
    if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.time() - start_time
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Evaluation completed in {elapsed:.2f} seconds")

    if use_ddp and dist.is_initialized():   
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 



  

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