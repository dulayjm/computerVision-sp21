import numpy as np
import cv2
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans
import pickle

def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros ((alpha * len(imgPaths), 3 * len(filterBank)))
    responseMatrix = []

    for i, path in enumerate(imgPaths):
        print ('-- processing %d/%d' % (i, len(imgPaths)))
        img = cv2.imread ('../data/%s' % path)
        # should be OK in standard BGR format
        
        # -----fill in your implementation here --------
        responses = np.array([cv2.filter2D(img, -1, k) for k in filterBank])
        responses = extract_filter_responses(img, filterBank)
        for response in responses:
            points = []
            if method == 'Random':
                points = get_random_points(response, alpha)
            elif method == 'Harris':
                points = get_harris_points(response, alpha, 0.04)

            # print("points shape", points.shape)
            responseMatrix.extend(points)


            # responseMatrix = np.asarray(responseMatrix)

            # print("response Matrix shape", responseMatrix.shape)



    print('len of responseMatrix', len(responseMatrix))
    responseMatrix = np.asarray(responseMatrix)
    print('response matrix shape', responseMatrix.shape)

    # ----------------------------------------------

    # can use either of these K-Means approaches...  (i.e. delete the other one)
    # OpenCV K-Means
    #    pixelResponses = np.float32 (pixelResponses)
    #    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #    ret,label,dictionary=cv2.kmeans(pixelResponses,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # sklearn K-Means
    dictionary = KMeans(n_clusters=K, random_state=0).fit(responseMatrix).cluster_centers_
    return dictionary    


if __name__ == '__main__':
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))

    train_imagenames = meta['train_imagenames']

    # -----fill in your implementation here --------
    # dictionary = get_dictionary(train_imagenames, 50, 50, "Random")
    # with open('randomWords.pkl', 'wb') as handle:
    #     pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    otherDictionary = get_dictionary(train_imagenames, 50, 50, "Harris")
    with open('harrisWords.pkl', 'wb') as handle:
        pickle.dump(otherDictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

