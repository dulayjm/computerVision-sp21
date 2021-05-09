import numpy as np
import cv2
import random


def get_random_points (img, alpha):

    random.seed()
    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------
    x_len = img.shape[0]
    y_len = img.shape[1]
    # print('inside random points')
    # print('xlen', x_len)
    # print('ylen', y_len)
    minRanPoint = min(x_len, y_len)

    rows, cols = (alpha, 2)
    points=[]
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(random.randint(minRanPoint, minRanPoint))
        points.append(col)

    points = np.asarray(points)
    # print('points shape', points.shape)
    # ----------------------------------------------
    
    return points


# start of some code for testing get_random_points()
if __name__ == "__main__":
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    print (get_random_points (img, 50))

