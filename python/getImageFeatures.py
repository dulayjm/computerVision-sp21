import numpy as np


def get_image_features (wordMap, dictionarySize):

    # -----fill in your implementation here --------
    print('wordMap shape', wordMap.shape)
    # h = np.zeros(dictionarySize)
    # for i in wordMap: 
    h = np.histogram(wordMap, dictionarySize)
    h = h[0]
    print('h', len(h))

    # ----------------------------------------------
    return h
