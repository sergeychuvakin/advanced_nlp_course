import json
import sys

import torch
from config import Config, LanguageModelConfig
from datahandler import LMDataset
from dependencies import corpus, tokenizer
from loguru import logger
from model import LM_LSTM
from processing_utils import (clean_text, create_ngrams, create_to_x_and_y,
                              create_vocab, save_artifacts, split_on_sequences,
                              word2int)
from torch.utils.data import DataLoader
from train_utils import train_model

config = Config()

logger.remove()
logger.add(sys.stderr, level="WARNING")


corpus = clean_text(corpus)
corpus = split_on_sequences(corpus)

tcorpus = tokenizer.encode_batch(corpus)

## create n-grams for each doc
sq = create_ngrams(tcorpus, config.N_GRAM)

## shift corpus to create x and y
x, y = create_to_x_and_y(sq)

id_token, token_id = create_vocab(tokenizer)
vocab_size = len(token_id)

## split data
tradeoff_index = int(len(x) * config.TRAIN_PROPORTION)

x_train = x[:tradeoff_index]
x_test = x[tradeoff_index:]

y_train = y[:tradeoff_index]
y_test = y[tradeoff_index:]

logger.warning(
    f"""
    Output shapes: 
        x_train: {len(x_train)}, 
        x_test: {len(x_test)}, 
        y_train: {len(y_train)}, 
        y_test: {len(y_test)}
    """
)

## load to dataset and dataloader
train_ds = LMDataset(x_train, y_train)
test_ds = LMDataset(x_test, y_test)

train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# model and model config
model_config = LanguageModelConfig(vocab_size=vocab_size, emb_size=300)
model = LM_LSTM(**model_config.dict(), logger=logger)

## save artifacts
save_artifacts(
    (model_config.dict(), config.SAVE_MODEL_CONFIG),
    (token_id, config.SAVE_TOKEN_ID),
    (id_token, config.SAVE_ID_TOKEN),
)

optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)
loss_func = torch.nn.CrossEntropyLoss()


# train model
tmodel = train_model(
    model, train_dl, optimizer=optimizer, loss_func=loss_func, epochs=5, clip=1
)

torch.save(model.state_dict(), config.SAVE_MODEL_FNAME)


## validation
logger.warning(
    """
    Cross-Entropy: %f
    Perpelxity: %f
    """
    % val_metrics(model, test_dl, token_id)
)

## inference

from inference import (id_token, model, predict_one_word, predict_sample,
                       token_id)

predict_one_word("one of the", model, token_id, id_token, random=False)

predict_sample(
    "one of the", model, token_id, id_token, length_of_sample=10, random=True
)
