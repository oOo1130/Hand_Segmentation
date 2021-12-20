import cv2
import numpy as np
import sys
import os
angles = range(-2,3)
shifts = [[0,0],[0,1],[1,0],[1,1],[0,2],[2,0],[1,2],[2,1],[2,2],
                [0,-1],[-1,0],[-1,-1],[0,-2],[-2,0],[-1,-2],[-2,-1],[-2,-2],
                [1,-1],[1,-2],[2,-1],[2,-2],
                [-1,1],[-1,2],[-2,1],[-2,2]]
multiplier = len(angles)*len(shifts)
X_train=np.zeros((100*multiplier,128,128,3))
y_train=np.zeros((100*multiplier,128,128,3))
print(X_train.shape)
path_x = 'Data/X/'
path_y = 'Data/Y2/'
total = 0

#for pos in range(len(path_x)):
for img in sorted(os.listdir(path_x))[:-5]:
    image_x = cv2.imread(path_x+img, cv2.IMREAD_UNCHANGED)
    print(image_x.shape)
    dim = (128,128)
    originalIm = cv2.resize(image_x, dim, interpolation = cv2.INTER_AREA)
    print(originalIm.shape)
    image_y = cv2.imread(path_y+img, cv2.IMREAD_UNCHANGED)
    segmentedIm = cv2.resize(image_y, dim, interpolation = cv2.INTER_AREA)

    for angle in angles:
        for shift in shifts :

            M = cv2.getRotationMatrix2D((128/2,128/2),angle,1)
            shiftM = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
            rotatedIm = cv2.warpAffine(originalIm,M,(128,128))
            rotatedSegmentedIm = cv2.warpAffine(segmentedIm,M,(128,128))
            rotatedShiftedIm = cv2.warpAffine(rotatedIm,shiftM,(128,128))
            rotatedSegmentedShiftedIm = cv2.warpAffine(rotatedSegmentedIm,shiftM,(128,128))
            print(rotatedShiftedIm.shape)
            X_train[total]=rotatedShiftedIm
            y_train[total]=rotatedSegmentedShiftedIm
            total+=1
