import gc
import os
import torch
from bert_score import score as bertscore
from model_architecture.model import LanguageModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

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

def decode_ids(tokenizer, ids, stop_at_eos=True):
    """Decode token IDs to text"""
    text = tokenizer.decode(ids, skip_special_tokens=True)  # Skip special tokens by default
    
    if stop_at_eos:
        # Stop at EOS token
        if "[EOS]" in text:
            text = text.split("[EOS]")[0].strip()
    return text

def load_model(model_path,config):
    """Load a PCTransformer model from a checkpoint file."""
    model = LanguageModel(config)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded successfully from {model_path}")
    return model
