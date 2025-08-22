import torch
import gc
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bertscore
from tokenizers import Tokenizer
from training import TOKENIZER_DIR
from torch.nn.utils.rnn import pad_sequence


def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
def compute_text_metrics(predictions, targets):
    print("\nComputing BERTScore and BLEU...")
    P, R, F1 = bertscore(
        predictions,
        targets,
        lang="en",
        model_type="roberta-base",
        rescale_with_baseline=True,
    )
    print(f"BERTScore (F1): {F1.mean().item():.4f}")

    smooth_fn = SmoothingFunction().method4
    tokenized_targets = [[target.split()] for target in targets]
    tokenized_pred = [pred.split() for pred in predictions]
    bleu = corpus_bleu(tokenized_targets, tokenized_pred, smoothing_function=smooth_fn)
    print(f"BLEU Score: {bleu:.4f}")
    
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