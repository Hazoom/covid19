import string

import networkx as nx

from nlp.text_parsing import parse_text

WINDOW_SIZE = 3

POS_KEPT = ["ADJ",
            "NOUN",
            "PROPN",
            "VERB"]


def increment_edge(graph, node0, node1):
    """Increment the weight of the edge <``node0``, ``node1``> in ``graph``.

    Args:
        graph (nx.Graph): graph.
        node0 (int): first node id.
        node1 (int): second node id.
    """
    if graph.has_edge(node0, node1):
        graph[node0][node1]["weight"] += 1.0
    else:
        graph.add_edge(node0, node1, weight=1.0)


def link_sentence(doc, lemma_graph, seen_lemma):
    """Link a sentence in the ``lemma_graph`` by adding the relevant vertices and edges.

    Args:
        doc (spacy.Doc): sent.
        lemma_graph (nx.Graph): lemma graph.
        seen_lemma (dict): a vocab-dict for lemma.
    """
    visited_tokens = []
    visited_nodes = []

    for token in doc:
        if token.pos_ in POS_KEPT:
            key = (token.lemma_, token.pos_)
            if key not in seen_lemma:
                seen_lemma[key] = {token.lower}
            else:
                seen_lemma[key].add(token.lower)

            node_id = list(seen_lemma.keys()).index(key)
            if node_id not in lemma_graph:
                lemma_graph.add_node(node_id)

            for prev_token in range(len(visited_tokens) - 1, -1, -1):
                if (token.i - visited_tokens[prev_token]) <= WINDOW_SIZE:
                    increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
                else:
                    break

            visited_tokens.append(token.i)
            visited_nodes.append(node_id)


def get_labels(seen_lemma, non_lemma=False):
    """Get labels for seen lemmas.

    Args:
        seen_lemma (dict): map graph nodes to lemmas.
        non_lemma (bool): whether to include non lemmas.

    Returns:
        dict
    """
    labels = {}
    for node_id, lbl in enumerate(seen_lemma.keys()):
        lemma_, _ = lbl
        lemma_ = lemma_.lower()
        if non_lemma is False and lemma_ in string.punctuation:
            continue
        labels[node_id] = lemma_
    return labels


def collect_phrases(ranks, labels, num_phrases=5):
    """Collect phrases from a ranked lemma graph.

    Args:
        ranks (dict): map graph nodes to their ranking.
        lables (dict): map graph nodes to their corresponding lemma.
        num_phrases (int): num of phrases to return.

    Returns:
        list
    """
    phrase_list = []
    phrase_num = 0
    for node_id, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
        lemma = labels.get(node_id)
        if not lemma:
            continue
        phrase_list.append((lemma, rank))
        phrase_num += 1
        if phrase_num == num_phrases:
            break
    return phrase_list


def calc_textrank(corpus, num_phrases=5):
    """Build a lemma graph and calculate ``pagerank``.

    Args:
        corpus (list[str]): corpus of cleaned sentences.
        num_phrases (int): num of phrases to return.

    Returns:
        (lemma_graph, labels, ranks, phrase_list)
            s.t
                lemma_graph: nx.Graph
                labels: dict
                ranks: dict
                phrase_list: list[tuple]
    """
    lemma_graph = nx.Graph()
    seen_lemma = {}

    for sent in corpus:
        doc = parse_text(sent)
        link_sentence(doc, lemma_graph, seen_lemma)

    ranks = nx.pagerank(lemma_graph)
    labels = get_labels(seen_lemma)
    phrase_list = collect_phrases(ranks, labels, num_phrases=num_phrases)
    return lemma_graph, labels, ranks, phrase_list
