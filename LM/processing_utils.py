import json
import re
from itertools import chain
from typing import Any, Dict, Tuple, Union

from config import Config
from nltk import ngrams
from nltk.corpus import stopwords
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm



def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[\:\;\&\%\$\@\^\(\)\[\]\,\-\']", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{3}", ".", text)
    text = re.sub(r"[\.\?\!]", f"{Config.TOKEN_END_OF_SENTENCE}", text)
    text = text.strip()
    text = text.lower()
    text = re.sub(r"(?:^| )\w(?:$| |<eos>)", "", text)
    return text

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    return " ".join([word for word in tqdm(text.split()) if word not in stop])

def remove_speakers(text):
    return re.sub(r".+:\n", " ", text)


def split_on_sequences(corpus):
    return [seq.strip() for seq in corpus.split(Config.TOKEN_END_OF_SENTENCE)]


def create_ngrams(tokens_lists, N):
    return [tuple(ngrams(sent.ids, N)) for sent in tqdm(tokens_lists)]

def create_ngrams1(tokens_lists, N, vocab):
    
    tokens_lists = [[vocab[word] for word in sent] for sent in tqdm(tokens_lists)]
    
    return [tuple(ngrams(sent, N)) for sent in tqdm(tokens_lists)]


def create_to_x_and_y(tokens_grams: tuple):
    """
    Create y by shifting x.
    """
    return zip(*chain(*[zip(x, x[1:]) for x in tokens_grams]))


def create_vocab(
    tokenizer: BertWordPieceTokenizer,
) -> Tuple[Union[Dict[str, int], Dict[int, str]]]:

    decoder = tokenizer.get_vocab()
    decoder[len(decoder)] = Config.TOKEN_PADDING
    encoder = {ii: i for i, ii in decoder.items()}

    return encoder, decoder


def word2int(seq, token_id):
    return [token_id[i] for i in seq.split()]


def save_artifacts(*artifacts: Tuple[Tuple[Union[Any, str]]]) -> None:
    for arti in artifacts:
        with open(arti[1], "w") as f:
            json.dump(arti[0], f)


def load_artifact(fname: str):
    with open(fname, "r") as f:
        arti = json.load(f)
    return arti
