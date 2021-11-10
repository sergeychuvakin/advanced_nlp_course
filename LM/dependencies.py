import requests
from tokenizers import BertWordPieceTokenizer
import os
from config import Config

os.system("wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")

corpus = requests.get(Config.CORPUS_URL).text

tokenizer = BertWordPieceTokenizer(Config.TOKENIZER, 
                                   lowercase=True, 
                                   strip_accents=False, ## акцанты над буквами
                                   wordpieces_prefix="##")