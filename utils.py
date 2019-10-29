def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq
