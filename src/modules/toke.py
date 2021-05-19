from wand.image import Image
from wand.display import display
import cv2
import matplotlib.pyplot as plt
# file_name = r"D:\Desktop\PBE\Images\12out.png"
# file_name = r"D:\Desktop\PBE\Images\33.jpg"
# file_name = r"D:\Desktop\PBE\Images\D2D_2.png"


# print(COLORSPACE_TYPES)
def img_prcess(file_name):
    img = Image(filename=file_name, colorspace='gray')
    # img.adaptive_threshold(1000, 2000, 90)
    # img.resize(700, 976)
    img.alpha_channel = 'remove'
    img.auto_threshold('otsu')
    img.auto_orient()
    img.deskew(10)

    img.enhance()
    # img.trim()
    img.save(filename='abc.png')
    i = cv2.imread('abc.png')
    return i