from remove_lines import remove_line
import cv2
import numpy as np
import pytesseract
from scipy.ndimage import rotate
import re
from spellchecker import SpellChecker
from PIL import Image
import pandas as pd
import csv
import io
from toke import img_prcess
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def pre_processing(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    #image_original = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
    #image_original = cv2.resize(image_original, (1700, 2200))
    #image_original = cv2.resize(image_original, None ,fx=0.7, fy=0.7, interpolation= cv2.INTER_AREA)
    image_scaled = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    #### quay anh > 90
    rotate_img = image_scaled
    newdata=pytesseract.image_to_osd(image_scaled)
    angle= 360 - int(re.search('(?<=Rotate: )\d+', newdata).group(0))
    if angle > 0 and  angle < 360 :
        rotate_img = rotate(image_scaled, angle)
    # convert the image to grayscale
    gray = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
    # threshold the image after Gaussian filtering
    # medBlur = cv2.medianBlur(gray, 3)
    # gauBlur = cv2.GaussianBlur(medBlur, (3,3), 10)
    #return bit
    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #dilate = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

# file_path = r"D:\Desktop\PBE\D2D_2.png"
file_path = r"D:\Desktop\PBE\Images\m.png"

image = cv2.imread(file_path)
# img = deskewing(img)
# img = border(img)
image = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)


# line_removed = img_prcess(file_path)
# cv2.imwrite("temp.png", line_removed)
#file_path = "temp.png"
# img = pre_processing(line_removed)

data = pytesseract.image_to_string(image, lang= 'vie_fast', config='--psm 10')

print(data)

# with open("result1.txt", "w+", encoding="utf-8") as f:
#     f.write(data)
# f.close()
