import numpy as np
import cv2
from remove_lines import remove_line


#img1_path = '33.jpg'
#img1_path = r"D:\Desktop\PBE\VB GUI IT3_SO HOA_SO TTTT QBH-2.png"
img1_path = r"D:\Desktop\PBE\D2D_2.png"
image = cv2.imread(img1_path)
#image1 = remove_line(image)
img1 = cv2.resize(image, (1653, 2339))
# kernel = cv2.getStructuringElement(cv2.MORPH_ER, (3, 3))
# erode = cv2.erode(img1, kernel, iterations=1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray1, 90, 255, cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 80))
dilate = cv2.dilate(thresh, kernel, iterations=1)

contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if x + w > 0:
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 90, 255), 2)



cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.imshow('1', img1)
cv2.waitKey(0)



























# def adaptiveThresh(image, s, t):
#     h, w = image.shape
#     intImage = np.zeros((2339, 1653))
#     intImg = np.transpose(intImage)
#     img = np.transpose(image)
#     for i in range(0, w):
#         Sum = 0
#         for j in range(0, h):
#             Sum = Sum + img[i, j]
#             if i == 0:
#                 intImg[i, j] = Sum
#             else:
#                 intImg[i, j] = intImg[i - 1, j] + Sum

#     out = np.zeros((1653, 2339))
#     for i in range(0, w):
#         for j in range(0, h):
#             x1 = max(0, int(i - s/2))
#             x2 = min(1652, int(i + s/2))
#             y1 = max(0,int(j - s/2))
#             y2 = min(2338, int(j + s/2))
#             count = (x2 - x1)*(y2 - y1)
#             Sum = intImg[x2, y2] - intImg[x2, y1 -1] - intImg[x1 -1, y2] + intImg[x1 - 1, y1 - 1]
#             if img[i,j]*count <= (Sum*(100 - t)/100):
#                 out[i, j] = 255
#             else:
#                 out[i,j] = 0
#     return np.transpose(out)

# s = max(gray1.shape[0], gray1.shape[1])/8
# threshed = adaptiveThresh(gray1, s, t=0.15)