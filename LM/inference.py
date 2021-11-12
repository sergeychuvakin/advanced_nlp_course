from random import sample
from typing import Dict

import torch
from config import Config, LanguageModelConfig
from dependencies import tokenizer
from loguru import logger
from model import LM_LSTM
from processing_utils import load_artifact


def _transform_raw_word(word: str, token_id: Dict[str, int]) -> torch.tensor:
    int_word = token_id.get(word, token_id.get(Config.TOKEN_UNKNOWN))
    return torch.tensor([[int_word]]).to(device)


def _get_model_output(model, input_tensor, hidden_state=None):
    if not hidden_state:
        return model(input_tensor, model.init_state(1))
    else:
        return model(input_tensor, hidden_state)


def _transform_model_output(out, random, id_token, top=3):

    if random:
        idx = sample(list(torch.topk(out, 3).indices[0]), 1)[0].item()

    else:
        idx = torch.topk(out, top).indices[0][0].item()
    return id_token[str(idx)]


def predict_one_word(word, model, token_id, id_token, random=True):

    input_tensor = _transform_raw_word(word, token_id)
    out, h = _get_model_output(model, input_tensor)
    return _transform_model_output(out, random, id_token)


def predict_sample(word, model, token_id, id_token, length_of_sample, random=True):
    result = []
    h = None
    while len(result) < length_of_sample:

        input_tensor = _transform_raw_word(word, token_id)
        word, h = _get_model_output(model, input_tensor, h)
        result.append(_transform_model_output(word, random, id_token))
    return " ".join(result)


## load artifacts
token_id = load_artifact(Config.SAVE_TOKEN_ID)
id_token = load_artifact(Config.SAVE_ID_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = LanguageModelConfig.parse_file(Config.SAVE_MODEL_CONFIG)

## load trained model
model = LM_LSTM(**model_config.dict(), logger=logger)
model.load_state_dict(torch.load(Config.SAVE_MODEL_FNAME, map_location=device))
model.eval()

if __name__ == "__main__":

    predict_one_word("one of the", model, token_id, id_token, random=False)

    predict_sample("one", model, token_id, id_token, 10, random=True)
