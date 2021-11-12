import torch
from config import Config
from processing_utils import load_artifact
from typing import Dict

token_id = load_artifact(Config.SAVE_TOKEN_ID)
id_token = load_artifact(Config.SAVE_ID_TOKEN)


def transform_raw_word(word:str, token_id:Dict[str, int]) -> torch.tensor:
    int_word = token_id.get(word, token_id.get(Config.TOKEN_UNKNOWN))
    return torch.tensor([[int_word]])