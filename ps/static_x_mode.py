#! /usr/bin/env python3


import copy


def get_expected_time_between_updates(wrk_iter_times, x=None):
    wits = list(copy.deepcopy(wrk_iter_times))
    assert x is None or x >= 1 and x <= len(wits)
    wits.sort()

    if x is None or x == len(wits):
        return wits[-1]
    elif x == 1:
        epct_t = 1 / sum([1 / x for x in wits])
        return epct_t

    wits = [[x, x] for x in wits]

    update_intvls = []
    for _ in range(1000):
        intvl = wits[x - 1][0]
        update_intvls.append(intvl)
        for i in range(0, x):
            wits[i][0] = wits[i][1]
        for i in range(x, len(wits)):
            wits[i][0] -= intvl
        wits.sort(key=lambda x: x[0])

    return sum(update_intvls) / len(update_intvls)
