def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq


class MergeDict(dict):
    class Merge(object):
        def __init__(self, data):
            self.dict = data

        def __setitem__(self, name, other):
            for k1 in other:
                if k1 not in self.dict:
                    self.dict[k1] = {}

                for k2 in other[k1]:
                    self.dict[k1]['{}/{}'.format(name, k2)] = other[k1][k2]

    @property
    def merge(self):
        return self.Merge(self)
   

class Etcetera(object):
    def __init__(self, spectras=None, weights=None):
        if weights is None:
            weights = {}

        self.spectras = spectras
        self.weights = weights

    def __setitem__(self, name, other):
        assert other.spectras is None

        self.weights = {
            **self.weights,
            **{'{}/{}'.format(name, k): other.weights[k] for k in other.weights}
        }
