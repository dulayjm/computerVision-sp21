import numpy as np
import cv2


def get_harris_points (img, alpha, k):

    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------
    # cv2 wrapper function that finds our top Harris points
    # points = cv2.goodFeaturesToTrack(img,alpha,0.01,10)
    # points = np.int0(points)
    # print(len(points))
    # print('the shape of points', points.shape)



    dst = cv2.cornerHarris(img,2,3,k)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    #find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    #define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
    corners = corners[:alpha]
    # ----------------------------------------------
    
    # print('len of corners', len(corners))

    return corners


# start of some code for testing get_harris_points()
if __name__ == "__main__":
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    print (get_harris_points (img, 50, 0.04))
