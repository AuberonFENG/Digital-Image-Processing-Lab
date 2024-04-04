# file directory
# D:\DIP\FLANN_mapping.py

# this is file for FLANN mapping
import numpy as np
import cv2
from matplotlib import pyplot as plt
# this is the pattern image
'''
#queryImage = cv2.imread(r'D:\DIP\DIP\PGM_IMAGES\PGM_IMAGES\bridge.pgm',0)
queryImage = cv2.imread(r'E:\OLD_F\fyp\match\result_open73_original\9_result.pgm',cv2.CV_8UC1)
plt.imshow(queryImage,), plt.show()




# next is 8bit mapping test
#queryImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Open 0073_33600 8bit_value.pgm',cv2.CV_8UC1)
queryImage = cv2.imread(r'E:\OLD_F\fyp\match\result_open73_original_8bit\9_result_8bit.pgm',cv2.CV_8UC1)
trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Open 0073_33600 8bit_value.pgm',cv2.CV_8UC1)
#trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Closed_33500 0134 8bit_value.pgm',cv2.CV_8UC1)


# this is the background image (bigger & contains the pattern)

trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Open 0073_33600 16.pgm',cv2.CV_8UC1)
# trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Closed_33500 0134 16.pgm',cv2.CV_8UC1)

#plt.imshow(trainingImage,), plt.show()
'''

# this is test on partial open to open
# queryImage = cv2.imread(r'E:\OLD_F\fyp\match\result_open73_original_8bit\9_result_8bit.pgm',cv2.CV_8UC1)
# trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Open 0073_33600 8bit_value.pgm',cv2.CV_8UC1)


# this is test on open to close
# queryImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Open 0073_33600 8bit_value.pgm',cv2.CV_8UC1)
# trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Closed_33500 0134 8bit_value.pgm',cv2.CV_8UC1)

# this is test for qiqi
# queryImage = cv2.imread(r'E:\OLD_E\q2.jpg',cv2.CV_8UC1)
# trainingImage = cv2.imread(r'E:\OLD_E\q3.jpg',cv2.CV_8UC1)
queryImage = cv2.imread(r'D:\DIP\c1.png',cv2.CV_8UC1)
trainingImage = cv2.imread(r'D:\DIP\c2.png',cv2.CV_8UC1)

'''
# this is test from open to close  but upper part 
queryImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A_Open 0013 8bit_value.pgm',cv2.CV_8UC1)
trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\TMJ-A Closed - 0074 8bit_value.pgm',cv2.CV_8UC1)
'''
# creare SIFT and detect/compute
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(queryImage,None)
kp2, des2 = sift.detectAndCompute(trainingImage,None)

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)  #or pass empty dictionary

flann = cv2.FlannBasedMatcher(indexParams, searchParams)

matches = flann.knnMatch(des1,des2,k=2)

#prepare an empty mask to draw good matches
matchesMask = [[0,0] for i in range(len(matches))]

#David G. Lowe's ratio test, populate the mask
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        matchesMask[i] = [1,0]
        
drawParams = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)

resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)

plt.imshow(resultImage,), plt.show()
