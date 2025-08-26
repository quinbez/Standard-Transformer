import gc
import os
import torch
from bert_score import score as bertscore
from data_preparation.config import Config
from transformers import GPT2TokenizerFast
from torch.nn.utils.rnn import pad_sequence
from model_architecture.model import LanguageModel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from data_preparation.tokenized_set.tokenized_dataset import TokenizedDataset

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
    tokenizer_path = os.path.join(Config.TOKENIZER_DIR, f"gpt2_tokenizer_{Config.DATASET_NAME}.json")
    tokenizer= GPT2TokenizerFast.from_pretrained(tokenizer_path)
    special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(special_tokens)
    
    Config.VOCAB_SIZE = len(tokenizer) 
    Config.PAD_ID = tokenizer.pad_token_id
    Config.EOS_ID = tokenizer.eos_token_id
    return tokenizer

# def load_model(model_path,vocab_size):
#     """
#     Load a PCTransformer model from a checkpoint file.

#     Args:
#         model_path (str): Path to the saved model checkpoint.
#         config: Model configuration object.
#     Returns:
#         PCTransformer: The loaded model with weights.
#     """
#     model = LanguageModel(vocab_size)
#     model.load_state_dict(torch.load(model_path), strict = False)
#     return model
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


def get_datasets():
    train_dataset = TokenizedDataset("train", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
    valid_dataset = TokenizedDataset("valid", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
    test_dataset = TokenizedDataset("test", Config.TOKENIZER_DIR, Config.MAX_LENGTH)

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
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=Config.num_workers,
        pin_memory=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=Config.num_workers > 0,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=Config.num_workers > 0
    )                     
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=False,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=Config.num_workers > 0
    )
    return train_loader, valid_loader, test_loader
