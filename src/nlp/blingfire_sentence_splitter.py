import re
from blingfire import text_to_sentences
from nlp import common_sentence_splitter


PARAGRAPH_SEPARATOR_RE = re.compile(r'(?:\r?\n)+')


# This will entirely disable spaCy's sentence detection
def mark_sentence_boundaries(doc):
    text = doc.text
    sentences = split_text_to_sentences(text)
    return common_sentence_splitter.mark_sentence_boundaries_for_sentences(doc, sentences)


def split_text_to_sentences(text):
    paragraphs_result = []
    for paragraph in PARAGRAPH_SEPARATOR_RE.split(text):
        sentences = text_to_sentences(paragraph.strip()).split('\n')
        paragraphs_result.append([sentence.strip() for sentence in sentences])
    return paragraphs_result
