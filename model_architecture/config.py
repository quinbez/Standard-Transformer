from dataclasses import dataclass

# Global variables will be set in main() function
@dataclass
class GPTConfig:
    vocab_size: int
    pad_token_id: int 
    eos_token_id: int 
    block_size: int
    learning_rate: float
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    max_epochs: int
    max_new_tokens: int
    temperature: int