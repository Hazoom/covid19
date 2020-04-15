def get_encoder(name):
    if name == 'simple_encoder':
        from encoders.simple_encoder import SimpleEncoder  # pylint: disable=import-outside-toplevel
        encoder = SimpleEncoder.from_env()
    elif name == 'bert_encoder':
        from encoders.bert_encoder import BertEncoder
        return BertEncoder()
    else:
        raise ValueError(f'Unknown encoder: {name}')
    return encoder
