from torch.utils.data import DataLoader
from loguru import logger
import sys


from dependencies import corpus, tokenizer
from config import Config, LanguageModelConfig
from processing_utils import (
    clean_text, 
    split_on_sequences, 
    create_ngrams, 
    create_to_x_and_y, 
    word2int,
    create_vocab
)
from model import LM_LSTM
from datahandler import LMDataset
from train_utils import train_model

config = Config()

logger.remove()
logger.add(sys.stderr, level="WARNING")

corpus = clean_text(corpus)
corpus = split_on_sequences(corpus)

tcorpus = tuple(map(lambda sentence: tokenizer.encode(sentence), corpus))


## create n-grams for each doc
sq = create_ngrams(tcorpus, config.N_GRAM)

## shift corpus to create x and y 
x, y =  create_to_x_and_y(sq)

token_id, id_token = create_vocab(tcorpus)
vocab_size = len(token_id)

## for passing to dataloader
x_int = [word2int(i, token_id) for i in x]
y_int = [word2int(i, token_id) for i in y]

## split data
tradeoff_index = int(len(x_int) * config.TRAIN_PROPORTION)

x_train = x_int[:tradeoff_index]
x_test = x_int[tradeoff_index:]

y_train = y_int[:tradeoff_index]
y_test = y_int[tradeoff_index:]

logger.warning(f"Outpur shapes: x_train: {len(x_train)}, x_test: {len(x_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")

## load to dataset and dataloader
train_ds = LMDataset(x_train, y_train)
test_ds = LMDataset(x_test, y_test)

train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# model and model config
model_config = LanguageModelConfig(vocab_size=vocab_size, emb_size=300)
model = LM_LSTM(**model_config.dict(), logger=logger)


optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)
loss_func = torch.nn.CrossEntropyLoss()

# train model 
tmodel = train_model(model,
                     train_dl,
                     optimizer=optimizer,
                     loss_func=loss_func,
                     batch_size=config.BATCH_SIZE,
                     epochs=30, 
                     clip=1)

# torch.save(model.state_dict(), config.SAVE_MODEL_FNAME)