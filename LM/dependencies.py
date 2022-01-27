import os

import requests
from config import Config
from tokenizers import BertWordPieceTokenizer

os.system(
    "wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
)

corpus = requests.get(Config.CORPUS_URL).text

tokenizer1 = BertWordPieceTokenizer(
    Config.TOKENIZER,
    lowercase=True,
    strip_accents=False,  ## акцанты над буквами
    wordpieces_prefix="##",
)

from nltk.tokenize import word_tokenize

tokenizer = word_tokenize