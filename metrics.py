import numpy as np


def edit_distance(ref, hyp):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    dist = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint8).reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                dist[0][j] = j
            elif j == 0:
                dist[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                substitute = dist[i - 1][j - 1] + 1
                insert = dist[i][j - 1] + 1
                delete = dist[i - 1][j] + 1
                dist[i][j] = min(substitute, insert, delete)
    return dist


def get_step_list(ref, hyp, dist):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(ref)
    y = len(hyp)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and dist[x][y] == dist[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and dist[x][y] == dist[x][y - 1] + 1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and dist[x][y] == dist[x - 1][y - 1] + 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def word_error_rate(ref, hyp):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    dist = edit_distance(ref, hyp)

    # find out the manipulation steps
    # list = get_step_list(ref, hyp, dist)

    # print the result in aligned way
    result = float(dist[len(ref)][len(hyp)]) / len(ref) * 100
    # result = str("%.2f" % result) + "%"
    # alignedPrint(list, r, h, result)

    return result
