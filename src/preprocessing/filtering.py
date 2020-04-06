def get_tags(sections):
    """
    Credit: https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings

    Searches input sections for matching keywords. If found, returns the keyword tag.
    Args:
        sections: list of text sections
    Returns:
        tags
    """

    keywords = ["2019-ncov", "2019 novel coronavirus", "coronavirus 2019", "coronavirus disease 19", "covid-19",
                "covid 19", "ncov-2019",
                "sars-cov-2", "wuhan coronavirus", "wuhan pneumonia", "wuhan virus"]

    tags = None
    for text in sections:
        if any(x in text.lower() for x in keywords):
            tags = "COVID-19"

    return tags
