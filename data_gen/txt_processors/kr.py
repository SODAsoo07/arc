import re
import unicodedata
import nltk

from g2pk import G2p
from g2pk.numerals import convert_num as normalize_numbers

from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from data_gen.txt_processors.base_text_processor import (
    BaseTxtProcessor,
    register_txt_processors,
)
from utils.text.text_encoder import PUNCS


class KrG2p(G2p):
    word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text):
        # preprocessing
        words = KrG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        return text[:-1]


@register_txt_processors("kr")
class TxtProcessor(BaseTxtProcessor):
    g2p = KrG2p()

    @staticmethod
    def preprocess_text(text):
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )  # Strip accents
        text = text.lower()
        text = re.sub("['\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ a-z{PUNCS}]", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = text.replace("i.e.", "that is")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        return text

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls.g2p(txt)
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == " ":
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt
