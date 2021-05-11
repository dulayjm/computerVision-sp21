import pickle
from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import cv2
import numpy as np
import matplotlib as plt

meta = pickle.load (open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']
testLabels = meta['test_labels']

# -----fill in your implementation here --------
meta = pickle.load(open('visionRandom.pkl', 'rb'))
trainHist = meta['trainFeatures']
trainLabels = meta['trainLabels']
filterBank = meta['filterBank']
randomWordsDictionary = meta['dictionary']

all_accuracies = []
for k in range(40):
    testHist = []
    predLabels = []
    correct = 0
    for i in range(len(test_imagenames)): 
        img_name = test_imagenames[i]
        img = cv2.imread ('../data/%s' % img_name)
        wordMap = get_visual_words(img, randomWordsDictionary, filterBank)
        testFts = get_image_features(wordMap, len(randomWordsDictionary))
        testHist.append(testFts)

        # minDist = 100000000000
        kminDist = [10000000000 for _ in range(k)]
        predLabel = None
        min_train_label_indices = []
        for j in range(len(trainHist)):
            dst = get_image_distance(trainHist[j], testFts, method='chi')
            if len(kminDist) > 0: 
                if dst < max(kminDist): 
                    # minDist = dst
                    kminDist.append(dst)
                    kminDist.remove(min(kminDist))

                    # vote on most common object of train labels at 
                    # the indices of their nearest neighbors points 
                    min_train_label_indices.append(j)
                    most_common_index = max(set(min_train_label_indices), key=min_train_label_indices.count)
                    
                    predLabel = trainLabels[most_common_index]
            else: 
                # minDist = dst
                kminDist.append(dst)

                # vote on most common object of train labels at 
                # the indices of their nearest neighbors points 
                min_train_label_indices.append(j)
                most_common_index = max(set(min_train_label_indices), key=min_train_label_indices.count)
                
                predLabel = trainLabels[most_common_index]

        # okay now we have the predLabel for specific test input image
        predLabels.append(predLabel)

        # accuracy: compare the predicted label with the actual ith testLabel
        if predLabel == testLabels[i]:
            correct += 1
        print(predLabels)
        accuracy = (correct/len(testLabels))
        print('accuracy', accuracy)
        print('k', k)
        all_accuracies.append(accuracy)

print('all accuracies', all_accuracies)

plt.plot(40, all_accuracies)
plt.title('Accuracy with respect to k-value')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.savefig('graph.png')


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(testLabels, predLabels)
print(cf)

# # ----------------------------------------------