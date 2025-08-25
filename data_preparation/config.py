import os
class Config:
    VOCAB_SIZE = None
    PAD_ID=None
    EOS_ID=None
    MAX_LENGTH = 128
    DATASET_NAME = "ptb" 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, "datasets") 
    TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer", "outputs")  
    BATCH_SIZE = 32
    num_workers = 8