def get_encoder(name):
    if name == 'simple_encoder':
        from encoders.simple_encoder import SimpleEncoder  # pylint: disable=import-outside-toplevel
        encoder = SimpleEncoder.from_env()
    elif name == 'use_encoder':
        from encoders.universal_encoder import UniversalSentenceEncoder  # pylint: disable=import-outside-toplevel
        encoder = UniversalSentenceEncoder.from_env()
    elif name == 'bert_encoder':
        encoder = None
    else:
        raise ValueError(f'Unknown encoder: {name}')
    return encoder
