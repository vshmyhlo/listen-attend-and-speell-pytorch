import numpy as np


def edit_distance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def get_step_list(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def word_error_rate(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = edit_distance(r, h)

    # find out the manipulation steps
    list = get_step_list(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    # result = str("%.2f" % result) + "%"
    # alignedPrint(list, r, h, result)

    return result
