import cv2
import numpy as np
import  matplotlib.pyplot as plt
#import image
img_path = r"D:\Desktop\PBE\vd1.jpg"
img  = cv2.imread(img_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

medBlur = cv2.medianBlur(gray, 5)
gauBlur = cv2.GaussianBlur(medBlur, (5,5), 10)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
erode = cv2.erode(gray, kernel, iterations=1)

th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

binar = cv2.bitwise_not(gray)

(h, w) = binar.shape[:2]

sumCols = []
for j in range(h):
	col = binar[0:w, j:j+1]
	sumCols.append(np.sum(col))

#print(sumCols)

for i in range(0, len(sumCols)):
	if sumCols[i] > 5000:
		cv2.line(img, (0, i), (w, i), (255, 0, 255), 1)


cv2.imshow('marked areas', img)
cv2.waitKey(0)