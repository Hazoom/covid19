import string
import re
from typing import List
import ftfy
import contractions

from nlp.resources import (CURRENCIES,
                           RE_NUMBER,
                           RE_URL,
                           STOP_WORDS)


def clean_tokenized_sentence(tokens: List[str],
                             unicode_normalization="NFC",
                             unpack_contractions=False,
                             replace_currency_symbols=False,
                             remove_punct=True,
                             remove_numbers=False,
                             lowercase=True,
                             remove_urls=True,
                             remove_stop_words=True) -> str:
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]

    sentence = ' '.join(tokens)

    if unicode_normalization:
        sentence = ftfy.fix_text(sentence, normalization=unicode_normalization)

    if unpack_contractions:
        sentence = contractions.fix(sentence, slang=False)

    if replace_currency_symbols:
        for currency_sign, currency_tok in CURRENCIES.items():
            sentence = sentence.replace(currency_sign, f'{currency_tok} ')

    if remove_urls:
        sentence = RE_URL.sub('_URL_', sentence)

    if remove_punct:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # strip double spaces
    sentence = re.sub(r' +', ' ', sentence)

    if remove_numbers:
        sentence = RE_NUMBER.sub('_NUMBER_', sentence)

    if lowercase:
        sentence = sentence.lower()

    return sentence
