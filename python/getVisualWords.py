import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import cv2
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points

def get_visual_words (img, dictionary, filterBank):

    # -----fill in your implementation here --------
    responses = np.array([cv2.filter2D(img, -1, k) for k in filterBank])
    print('responses.shape', responses.shape)
    print('dictionary.shape', dictionary.shape)
    m = []

    for response in responses:
        points = []
        points = get_harris_points(response, 50, 0.04)
        m.extend(points) 
    m = np.asarray(m)
    print('m.shape', m.shape)

    wordMap = cdist(m, dictionary)
    print('the type of the word map is: ', type(wordMap))
    print('the word map is:', wordMap)
    print('the length of the wordMap is', len(wordMap))
    # ----------------------------------------------

    return wordMap

