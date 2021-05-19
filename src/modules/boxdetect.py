import numpy as np
import cv2

def sort_contours(cnts):
    boundingBoxs = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxs) = zip(*sorted(zip(cnts, boundingBoxs), key=lambda b:b[1][1], reverse=False))
    return (cnts, boundingBoxs)

#img_path = r"D:\Desktop\PBE\VB GUI IT3_SO HOA_SO TTTT QBH-2.png"
img_path = r"D:\Desktop\PBE\D2D_2.png"

img =cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
inv = 255 - thresh

kernel_length = int(np.array(img).shape[1]/128)
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

ver_line = cv2.erode(inv, ver_kernel, iterations= 3)
verticle_line = cv2.dilate(ver_line, kernel, iterations=3)

hori_line = cv2.erode(inv, hori_kernel, iterations= 3)
horizontal_line = cv2.dilate(hori_line, kernel, iterations=3)

alpha = 0.5
beta = 1 - alpha

img_bin = cv2.addWeighted(verticle_line, alpha, horizontal_line, beta, 0.0)
img_bin = cv2.dilate(img_bin, kernel, iterations=1)
thresh_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cnts, hier = cv2.findContours(thresh_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts, boundingBox = sort_contours(cnts)

cv2.drawContours(img, cnts, -1,color=(0, 255, 127), thickness= 2)
crops = []
# print(len(cnts))
# for c in cnts:
#     x , y, w, h = cv2.boundingRect(c)
#     crop = img[y:y+h, x:x+w]
#     crops.append(crop)
#     cv2.rectangle(img, (y, y+h), (x, x+w), (0, 255, 90), 2)


# cv2.namedWindow('1', cv2.WINDOW_NORMAL)
#show = cv2.resize(img, None, fx = 0.35, fy=0.35, interpolation= cv2.INTER_AREA)
cv2.imshow('1', img)
# for crop in crops:
#     cv2.imshow('1', crop)
#     cv2.waitKey(0)
cv2.waitKey(0)