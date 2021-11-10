import re
from nltk import ngrams
from itertools import chain
import torch
from tqdm import tqdm
from tokenizers import Encoding
from typing import Tuple, Union, Dict

from config import Config

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[\:\;\&\%\$\@\^\(\)\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.\?\!\.{3}]", f".{Config.TOKEN_END_OF_SENTENCE}", text)
    text = text.strip()
    text = text.lower()
    return text

def split_on_sequences(corpus):
    return corpus.split(Config.TOKEN_END_OF_SENTENCE)


def create_ngrams(tokens_lists, N):
    return [tuple(map(lambda y: " ".join(y), ngrams(sent.tokens, N))) for sent in tqdm(tokens_lists)]


def create_to_x_and_y(tokens_grams: tuple):
    """
    Create y by shifting x. 
    """
    return zip(*chain(*[zip(x, x[1:]) for x in tokens_grams])) 


def create_vocab(tcorpus: Tuple[Encoding]) -> Tuple[Union[Dict[str, int], Dict[int, str]]]:

    decoder = dict(enumerate(set(chain(*[sent.tokens for sent in tqdm(tcorpus)]))))
    decoder[len(decoder)] = Config.TOKEN_PADDING
    encoder = {ii:i for i, ii in decoder.items()}
    
    return encoder, decoder

def word2int(seq, token_id):
    return [token_id[i] for i in seq.split()]