from utils import chi2dist
import numpy as np
from scipy.spatial.distance import cdist


def get_image_distance(hist1, hist2, method='euclidean'):

    dist = 0
    if method == 'euclidean':
        dist = cdist(hist1, hist2)
    else:
        dist = chi2dist(hist1, hist2)

    return dist