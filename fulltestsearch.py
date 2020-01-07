import numpy as np
import cv2
import os
import pytesseract
from pre_processing import pre_processing
from timeit import default_timer as timer
from multiprocessing.pool import Pool
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = r"D:\Desktop\PBE\31.signed.pdf"

pages = convert_from_path(path, dpi=300, first_page=0,last_page=2)

# print(type(pages))


def ocr(page):
    img = pre_processing(page)
    data = pytesseract.image_to_string(img, lang='vie_fast')
    return data


#     data1 = pytesseract.image_to_string(img, lang='vie_fast')
#     return data1


if __name__ == "__main__":
    start = timer()
    path = r"D:\Desktop\PBE\31.signed.pdf"

    pages = convert_from_path(path, dpi=300, first_page=0,last_page=2)
    # with Pool(4) as p:
    #     data = p.map(ocr, pages)
    # print(data)
    strss = []
    for page in pages:
        data = ocr(page)
        strss.append(data)
    print(strss)
    # ocr(img_path1)
    # ocr(img_path2)
    # ocr(img_path3)
    print(timer()-start)
