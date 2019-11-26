class MergeDict(dict):
    class Merge(object):
        def __init__(self, data):
            self.dict = data

        def __setitem__(self, name, other):
            for k1 in other:
                if k1 not in self.dict:
                    self.dict[k1] = {}

                for k2 in other[k1]:
                    k_new = '{}/{}'.format(name, k2)
                    assert k_new not in self.dict[k1]
                    self.dict[k1][k_new] = other[k1][k2]

    @property
    def merge(self):
        return self.Merge(self)


def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq


def label_smoothing(input, smoothing):
    return input * (1 - smoothing) + smoothing / input.size(2)


def one_hot(input, num_classes):
    input = torch.eye(num_classes, dtype=torch.float, device=input.device)[input]

    return input
