import numpy as np
import cv2
from box_detect import box_extraction

def findctrs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 250))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return ctrs

img_path = r"D:\Desktop\PBE\Images\HSCT.010.18.00000129#0008-2.png"
img = cv2.imread(img_path)
table = box_extraction(img)
ctrs = findctrs(table)
# cv2.drawContours(img, ctrs, -1, (255, 12, 120), 3)
for c in ctrs:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(table, (x,y), (x+w, y+h), (127, 50, 90), 3)
cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.imshow('1', table)
cv2.waitKey(0)
cv2.destroyAllWindows()