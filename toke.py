import json
import numpy as np
import cv2

img_path = r'D:\Desktop\PBE\VB GUI IT3_SO HOA_SO TTTT QBH-2.png'
# img_path = r'D:\Desktop\PBE\1_6.png'
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((2,2), np.uint8)
img_dilation = cv2.erode(thresh, kernel, iterations=1)

# cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.imshow('1', img_dilation)
cv2.waitKey(0)