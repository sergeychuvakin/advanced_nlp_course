# import sys
# from loguru import logger
# from typing import Tuple, Union, Dict
from pydantic import BaseModel


class Config:
    CORPUS_URL: str = (
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
    )
    TOKENIZER: str = "bert-base-uncased-vocab.txt"
    N_GRAM: int = 3
    BATCH_SIZE: int = 1000
    TOKEN_END_OF_SENTENCE: str = "<eos>"
    TOKEN_PADDING: str = "[PAD]"
    TOKEN_UNKNOWN: str = "[UNK]"
    TRAIN_PROPORTION = 0.86
    SAVE_MODEL_FNAME = "myLM(1).pt"
    SAVE_MODEL_CONFIG: str = "model_config.json"
    SAVE_TOKEN_ID: str = "token_id.json"
    SAVE_ID_TOKEN: str = "id_token.json"


class LanguageModelConfig(BaseModel):
    vocab_size: int
    emb_size: int
    n_hidden: int = 256
    n_layers: int = 4
    drop_prob: float = 0.3
    lr: float = 0.001
