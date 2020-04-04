def mark_sentence_boundaries_for_sentences(doc, sentences):
    sentence_index = 0
    paragraph_index = 0
    index_in_sentence = 0

    for token in doc:
        token.is_sent_start = False

        if _need_to_skip(token, sentences, paragraph_index, sentence_index, index_in_sentence):
            continue

        try:
            potential_curr_index = sentences[paragraph_index][sentence_index].index(token.text, index_in_sentence)
        except ValueError:
            sentence_index += 1
            index_in_sentence = 0
            if len(sentences[paragraph_index]) <= sentence_index:
                sentence_index = 0
                while True:
                    paragraph_index += 1
                    if sentences[paragraph_index] and sentences[paragraph_index][sentence_index].strip():
                        break
            try:
                potential_curr_index = sentences[paragraph_index][sentence_index].index(token.text, index_in_sentence)
            except ValueError:
                # mismatch between spacy tokenization and sentence spliting tokenization
                token.is_sent_start = True
                index_in_sentence = _find_shared_start(token.text, sentences[paragraph_index][sentence_index])
                continue

        if index_in_sentence == 0 and sentences[paragraph_index][sentence_index].startswith(token.text):
            token.is_sent_start = True
        index_in_sentence = potential_curr_index + len(token.text)

    return doc


def _find_shared_start(token_text, sentence):
    shared_chars = 0
    max_shared_chars = 0
    while shared_chars < len(token_text):
        if sentence.startswith(token_text[len(token_text)-shared_chars:]):
            max_shared_chars = shared_chars
        shared_chars += 1

    return max_shared_chars


def _need_to_skip(token, sentences, paragraph_index, sentence_index, index_in_sentence):
    if _is_empty(token):
        return True

    return _is_in_sentences(index_in_sentence, paragraph_index, sentence_index, sentences, token)


def _is_empty(token):
    return not token.text.strip()


def _is_in_sentences(index_in_sentence, paragraph_index, sentence_index, sentences, token):
    sentence = sentences[paragraph_index][sentence_index]
    try:
        sentence.index(token.text, index_in_sentence)
        return False
    except ValueError:
        while True:
            sentence_index += 1
            if len(sentences[paragraph_index]) <= sentence_index:
                sentence_index = 0
                paragraph_index += 1
            if len(sentences) <= paragraph_index:
                return True
            if sentences[paragraph_index][sentence_index].strip():
                break
        sentence += sentences[paragraph_index][sentence_index]
        try:
            sentence.index(token.text, index_in_sentence)
            return False
        except ValueError:
            return True
