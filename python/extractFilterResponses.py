import cv2
import numpy as np
from utils import *
from createFilterBank import create_filterbank


def extract_filter_responses (img, filterBank):

    if len(img.shape) == 2:
        img = cv2.merge ([img, img, img])

    img = cv2.cvtColor (img, cv2.COLOR_BGR2Lab)
    filterResponses = []

    # -----fill in your implementation here --------
    images = np.array([cv2.filter2D(img, -1, k) for k in filterBank])
    for i, val in enumerate(images): 
        filename = '../q1-1_imgs/{}-.png'.format(i)
        cv2.imwrite(filename, val)
    return np.max(images, 0)
    # # ----------------------------------------------

    # return filterResponses

# start of some code for testing extract_filter_responses()
if __name__ == "__main__":
    fb = create_filterbank ()

    print("len of fb", len(fb))

    img = cv2.imread ("../data/football_stadium/sun_abcyqxcuxdpbmgkn.jpg")

#    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
#    print (extract_filter_responses (gray, fb))

    extract_filter_responses(img, fb)

