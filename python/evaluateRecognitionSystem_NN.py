import pickle
from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import cv2
import numpy as np


meta = pickle.load (open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']


# -----fill in your implementation here --------
# print(test_imagenames)
# okay, load in the train histogram 
# MAKE a test histogram 
# evaluate the two using the get_image_distance function

meta = pickle.load(open('/Users/justindulay/Downloads/scene_classification/python/visionRandom.pkl', 'rb'))
trainHist = meta['trainFeatures']
filterBank = meta['filterBank']
randomWordsDictionary = meta['dictionary']

testHist = None
for i in range(len(test_imagenames)): 
    img_name = test_imagenames[i]
    img = cv2.imread ('../data/%s' % img_name)
    wordMap = get_visual_words(img, randomWordsDictionary, filterBank)
    testHist = get_image_features(wordMap, len(randomWordsDictionary))

print('trainHist.shape', trainHist.shape)
print('testHist.shape', testHist.shape)

dist = get_image_distance(trainHist, testHist, 'euclidean')
print('dist', dist)
# ----------------------------------------------