from utils import MergeDict, take_until_token


def test_take_until_token():
    actual = take_until_token('hello world', ' ')

    assert actual == 'hello'


def test_merge_dict():
    d = MergeDict(k1=1)
    d.merge['name'] = MergeDict(k2={'key': 'value'})

    assert d['k1'] == 1
    assert d['k2']['name/key'] == 'value'
