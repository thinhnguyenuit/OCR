import cv2
import numpy as np
import pytesseract
import re
from pre_processing import pre_processing
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#img_path = r'D:\Desktop\PBE\Test\30.signed-page-001.jpg'
#img_path = r'D:\Desktop\PBE\Test\VanBanGoc_NQ-04.2019-về-án-lệ-1.jpg'
#img_path =  r"D:\Desktop\PBE\Test\img20190110_0002-1.jpg"
#img_path = r"D:\Desktop\PBE\Test\giay_chung_nhan_suc_khoe-1.jpg"
#img_path = r'D:\Desktop\PBE\Test\page4.png'
# img_path = r'D:\Desktop\PBE\temp.png'
#img_path = r'D:\Desktop\PBE\Test\img20190110_0005-1.jpg'
#img_path = r'D:\Desktop\PBE\Test\img20190110_0001-1.jpg'
#img_path = r"D:\OCR\2019-09-06-10-48-21.png"
#img_path = r"D:\OCR\input\chung chi it3_1.jpg"
#img_path = r"D:\OCR\input\2019-09-06-11-03-08.png"
# img_path = r'D:\Desktop\PBE\VB GUI IT3_SO HOA_SO TTTT QBH-2.png'
# img_path = r'D:\Desktop\PBE\1_6.png'
img_path =r"D:\Desktop\PBE\D2D_2.png"
img = pre_processing(img_path)


text = pytesseract.image_to_string(img, lang='vie_fast', config='psm -1')

cv2.imwrite("1.jpg", img)


with open("result.txt", "w+", encoding="utf-8") as f:
    f.write(text)

f.close()
