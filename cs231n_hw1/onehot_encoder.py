import numpy as np

def onehot_encoder(array, num_label):
    length = len(array)
    res = np.zeros((length, num_label))
    res[np.arange(length), array] = 1
    return res

