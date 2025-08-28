
# Standard Transformer

A standard transformer for text generation.


## Installation

Clone the Repository

```bash
git clone https://github.com/Betty987/Standard-Transformer.git
```
Navigate to the Project Directory

```bash
cd Standard-Transformer
``` 
Install Dependencies

Ensure you have a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``` 
Install required packages:

```bash
pip install -r requirements.txt
``` 
Tokenize data
```bash
python -m data_preparation.tokenizer.gpt2_tokenizer
``` 
Train the model
```bash
python training.py
``` 
