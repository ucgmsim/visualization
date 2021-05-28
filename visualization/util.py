import numpy as np

def intersection(lists):
    """
    Like np.intersect1d but between more than 2 lists.
    """
    common = lists[0]
    for items in lists[1:]:
        common = np.intersect1d(common, items)
    return common
