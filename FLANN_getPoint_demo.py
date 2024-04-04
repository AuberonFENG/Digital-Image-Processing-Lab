# this is file for FLANN mapping
import numpy as np
import cv2
from matplotlib import pyplot as plt

max_index_list = []

y = 0

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)  #or pass empty dictionary
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

queryImage = cv2.imread(r'E:\OLD_F\fyp\match\8bitOpen\TMJ-A_Open 0073_33600 8bit_value.pgm',cv2.CV_8UC1)
trainingImage = cv2.imread(r'E:\OLD_F\fyp\match\8bitClose\TMJ-A_Closed_33500 0134 8bit_value.pgm',cv2.CV_8UC1)

# for open the index from 030 to 111
open_imgs = []
m_open = 30
n_open = 111

num_images_open = n_open - m_open + 1
num_columns = 512
num_rows = 512

for i in range(m_open, n_open + 1):
    filename1 = r'E:\OLD_F\fyp\match\8bitOpen\TMJ-A_Open {:04d}_33600 8bit_value.pgm'.format(i)
    img = cv2.imread(filename1, cv2.CV_8UC1)
    open_imgs.append(img)

# for close the index from 065 to 249
close_imgs = []
p = 65
q = 249
num_images = q - p + 1
num_columns = 512
num_rows = 512

for j in range(p,q+1):
    filename2 = r'E:\OLD_F\fyp\match\8bitClose\TMJ-A_Closed_33500 {:04d} 8bit_value.pgm'.format(j)
    img = cv2.imread(filename2, cv2.CV_8UC1)
    close_imgs.append(img)# this is training image set

k_value = 26

sift = cv2.SIFT_create()
# sift = cv2.xfeatures2d()
open_point_list = []
closed_point_list = []

'''
for x in range(num_images_open):# num_images_open

    kp1, des1 = sift.detectAndCompute(open_imgs[x],None)
    

    # y is the yth close training image
    # creare SIFT and detect/compute
    # sift = cv2.SIFT_create()
    y = x + k_value 
    
    kp2, des2 = sift.detectAndCompute(close_imgs[y],None)
    
    matches=[]
    matches = flann.knnMatch(des1,des2,k=2)
    
    # print(len(matches))

    #prepare an empty mask to draw good matches
    matchesMask = [[0,0] for i in range(len(matches))]
    count_point = 0
    #David G. Lowe's ratio test, populate the mask
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
            pt1 = kp1[m.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            
            # if i % 5 ==0:
            #     cv2.circle(queryImage, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
            #     cv2.circle(trainingImage, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)
            # correction points
            diff = 6
            if (abs(int(pt1[0])-int(pt2[0])) < diff and abs(int(pt1[1])-int(pt2[1])) < diff):
                # abs(int(pt1[0])-int(pt2[0])) < diff
                # abs(int(pt1[1])-int(pt2[1])) < diff

                # this is specific points data
                # print(i, pt1, pt2)
                count_point=count_point+1
                
    print("image:",(x))
    print(" has matched points:",count_point)
    count_point_list.append(count_point)
    
    # drawParams = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
    # resultImage = cv2.drawMatchesKnn(open_imgs[x], kp1, close_imgs[y], kp2, matches, None, **drawParams)
    # plt.imshow(resultImage,), plt.show()
    
    # show one open image to all close
    # if (x == 9000):
    #     plt.scatter(range(num_images), count_point_list)
    #     plt.title("Histogram of matched points distribution: {}".format(x))
    #     plt.xlabel("close image id")
    #     plt.ylabel("number of points matched")
    #     plt.show()

    # #find the id of max count_point
    # max_index = count_point_list.index(max(count_point_list))
    # print(m_open+x, m_open+y, max_index)
    # max_index_list.append(max_index)
print(count_point_list)
'''
        



# drawParams = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)

# resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)

# plt.imshow(resultImage,), plt.show()


import pickle
import numpy as np

#point = ()


#check x,y
kp1, des1 = sift.detectAndCompute(queryImage,None)
    

# y is the yth close training image
# creare SIFT and detect/compute
# sift = cv2.SIFT_create()
#y = x + k_value 

kp2, des2 = sift.detectAndCompute(trainingImage,None)

matches=[]
matches = flann.knnMatch(des1,des2,k=2)
z_value = 43
a = []
array_point = []
# print(len(matches))

#prepare an empty mask to draw good matches
matchesMask = [[0,0] for i in range(len(matches))]
count_point = 0
#David G. Lowe's ratio test, populate the mask
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]
        pt1 = kp1[m.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
        pt2 = kp2[m.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
        # print(kpts1)
        # print(z_value)
        #a[z_value] = [z_value]
        # if i % 5 ==0:
        #     cv2.circle(queryImage, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
        #     cv2.circle(trainingImage, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)
        # correction points
        diff = 6
        
        if (abs(int(pt1[0])-int(pt2[0])) < diff and abs(int(pt1[1])-int(pt2[1])) < diff):
            # abs(int(pt1[0])-int(pt2[0])) < diff
            # abs(int(pt1[1])-int(pt2[1])) < diff

            # this is specific points data
            # print(i, pt1, pt2)
            #with open('point.pp','wb') as f:(pickle.dump(point,f))
            # count_point=count_point+1
            x1 = pt1[0]
            y1 = pt1[1]
            z1 = z_value

            x2 = pt2[0]
            y2 = pt2[1]
            z2 = z_value+k_value

            #z_value += 1

            point1 = (x1,y1,z1)# open point
            point2 = (x2,y2,z2)# close point
            open_point_list.append(point1)
            closed_point_list.append(point2)
            # print(i, point1, point2)
        #array_point[z_value][z_value] = [pt1[0],pt1[1]]
        #print[array_point]

        
print(open_point_list)
print("#############")
print(closed_point_list)






#array = np.hstack((pt1,a))
#print(array)

#drawParams = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)

#resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)

#plt.imshow(resultImage,), plt.show()