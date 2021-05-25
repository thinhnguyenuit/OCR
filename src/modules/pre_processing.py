import cv2
import os
import numpy as np
from scipy.ndimage import rotate
import pytesseract
import re


# img_path = r'D:\Desktop\PBE\thhaa.jpg'

def pre_processing(img):
    # image_original = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
    # image_original = cv2.resize(image_original, (1700, 2200))
    # image_original = cv2.resize(image_original, None ,fx=0.7, fy=0.7, interpolation= cv2.INTER_AREA)
    # image_scaled = cv2.resize(image_original, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # quay anh > 90
    rotate_img = img  # image_scaled
    newdata = pytesseract.image_to_osd(img)
    angle = 360 - int(re.search('(?<=Rotate: )\d+', newdata).group(0))
    if angle > 0 and angle < 360:
        rotate_img = rotate(image_scaled, angle)
    # convert the image to grayscale
    gray = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
    # threshold the image after Gaussian filtering
    # medBlur = cv2.medianBlur(gray, 3)
    # gauBlur = cv2.GaussianBlur(gray, (3,3), 10)
    # return bit
    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    medBlur = cv2.medianBlur(gray, 3)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # dilate = cv2.dilate(thresh, kernel, iterations=1)
    return medBlur
