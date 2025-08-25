# Hyperparameters

# Global variables will be set in main() function
class GPTConfig:
    vocab_size = None
    pad_token_id = None
    eos_token_id = None
    block_size = 256
    learning_rate = 1e-5
    n_embd = 64
    n_head = 2
    n_layer = 2
    dropout = 0.1
    max_epochs = 10
    max_new_tokens = 50
    temperature = 1.0