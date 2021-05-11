import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import cv2
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from skimage.color import label2rgb

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


    # print('word map type', wordMap.shape)
    # w = label2rgb(wordMap[0][0])
    # print('type of w', type(w))
    # im_bgr = cv2.cvtColor(w, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('../q2-3_imgs/img2.png', im_bgr)
    # ----------------------------------------------

    return wordMap