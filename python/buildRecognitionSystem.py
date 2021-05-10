import numpy as np
import cv2
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words
from sklearn.cluster import KMeans
import pickle


if __name__ == '__main__':
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    train_imagenames = meta['train_imagenames']
    trainLabels = meta['train_labels']

    randomWordsDictionary = pickle.load(open('randomWords.pkl', 'rb'))
    harrisWordsDictionary = pickle.load(open('harrisWords.pkl', 'rb'))

    filterBank = create_filterbank()

    # for random set 
    imageFeaturesRandom = []
    for i in range(len(train_imagenames)): 
        img_name = train_imagenames[i]
        img = cv2.imread ('../data/%s' % img_name)
        wordMap = get_visual_words(img, randomWordsDictionary, filterBank)
        fts = get_image_features(wordMap, len(randomWordsDictionary))
        imageFeaturesRandom.append(fts)

    # for harris set 
    imageFeaturesHarris = []
    # so you messed up here, you need to build the images for all of them
    for i in range(len(train_imagenames)):
        img_name = train_imagenames[i]
        img = cv2.imread ('../data/%s' % img_name) 
        wordMap = get_visual_words(img, harrisWordsDictionary, filterBank)
        fts = get_image_features(wordMap, len(harrisWordsDictionary))
        imageFeaturesHarris.append(fts)

    visionRandom = {
        'dictionary': randomWordsDictionary,
        'filterBank': filterBank,
        'trainFeatures': imageFeaturesRandom,
        'trainLabels': trainLabels
    }
    pickle.dump(visionRandom, open('visionRandom.pkl', 'wb'))

    visionHarris = {
        'dictionary': harrisWordsDictionary,
        'filterBank': filterBank,
        'trainFeatures': imageFeaturesHarris,
        'trainLabels': trainLabels
    }
    pickle.dump(visionHarris, open('visionHarris.pkl', 'wb'))