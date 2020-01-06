import pytesseract
import cv2

import numpy as np

def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxA_area + boxB_area - interArea)

    return iou


img1_path = '7.jpg'
img2_path = '3.jpg'
img3_path = '33.jpg'
img4_path = r'D:\OCR\jquery-select-areas-master\example\image.png'

image1 = cv2.imread(img1_path)
img1 = cv2.resize(image1, (1653, 2339))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread(img2_path)
img2 = cv2.resize(image2, (1653, 2339))

image3 = cv2.imread(img4_path)
img3 = cv2.resize(image3, (1653, 2339))

def toGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

boxes = []
boxx = (484, 175, 1140, 383, 1, 2)
# boxy = (250, 140, 580, 250, 1, 1)
# boxz = (650, 140, 1450, 250, 1, 3)
# boxa = (650, 280, 1450, 390, 1, 4)
boxes.append(boxx)
# boxes.append(boxy)
# boxes.append(boxz)
# boxes.append(boxa)

# p_1 = (boxx[0], boxx[1])
# p_2 = (boxx[2], boxx[3])
# p_11 = (boxy[0], boxy[1])
# p_12 = (boxy[2], boxy[3])
# p_21 = (boxz[0], boxz[1])
# p_22 = (boxz[2], boxz[3])
# p_31 = (boxa[0], boxa[1])
# p_32 = (boxa[2], boxa[3])

# cv2.rectangle(img3, p_1, p_2, (90, 0, 255), 3)
# cv2.rectangle(image1, p_11, p_12, (90, 0, 255), 3)
# cv2.rectangle(image1, p_21, p_22, (90, 0, 255), 3)
# cv2.rectangle(image1, p_31, p_32, (90, 0, 255), 3)
# cv2.namedWindow('1', cv2.WINDOW_NORMAL)
# cv2.imshow('1', img3)
# cv2.waitKey(0)

def findBoxContours(img, thresh_value, kernel_x, kernel_y):
    img = toGray(img)
    thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_x, kernel_y))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    print(len(contours))
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        box = (x, y, x + w, y + h)
        boxes.append(box)
    print(boxes)
    return boxes


def findRegion(boxO, img, iou_thresh, thresh_value, kernel_x, kernel_y):
    boxes = findBoxContours(img, thresh_value, kernel_x, kernel_y)
    rightBox = []
    for box in boxes:
        if IOU(boxO, box) > iou_thresh:
            #rightBox.append(box)
            RBox = (box[0], box[1], box[2], box[3], boxO[4], boxO[5])
            rightBox.append(RBox)
    return rightBox


def auto_detect(boxes, img, iou_thresh, thresh_value, kernel_x, kernel_y):
    pts =[]
    for boxx in boxes:
        draw_box = findRegion(boxx, img, iou_thresh, thresh_value, kernel_x, kernel_y)
        for pt in draw_box:
            #cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (90, 0, 255), 3)
            pts.append(pt)
    return pts

kernel_x = 100
kernel_y = 10

thresh_value = 90

iou_thresh = 0.1

pts = auto_detect(boxes, img3, iou_thresh, thresh_value, kernel_x, kernel_y)

for pt in pts:
    cv2.rectangle(img3, (pt[0], pt[1]), (pt[2], pt[3]), (0, 90, 255), 3)

ctr = findBoxContours(img3, thresh_value, kernel_x, kernel_y)
# print(ctr)

cv2.namedWindow('2', cv2.WINDOW_NORMAL)
cv2.imshow('2', img3)
cv2.waitKey(0)
