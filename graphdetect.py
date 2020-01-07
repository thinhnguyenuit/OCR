import matplotlib.pyplot as plt ; plt.rcdefaults()
import cv2
import numpy as np

def sort_contours(cnts):
    boundingBoxs = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxs) = zip(*sorted(zip(cnts, boundingBoxs), key=lambda b:b[1][0], reverse=False))
    return cnts

# img_path = r"D:\Desktop\PBE\area_graph.png"
img_path = r"D:\Desktop\PBE\Highlight-Data-Bar-Chart-3.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
med = cv2.medianBlur(gray, 15)
thresh = cv2.threshold(med, 127, 255, cv2.THRESH_BINARY_INV)[1]
# edges = cv2.Canny(thresh, 50, 150, 3)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
# lines = cv2.HoughLines(edges, 1, np.pi/100, 200)
# for line in lines:
#     for rho,theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))

#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# ctrs = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
# n_ctrs = sort_contours(ctrs)

# performance = []
# for ctr in n_ctrs:
#     if cv2.contourArea(ctr):
#         x, y, w, h = cv2.boundingRect(ctr)
#         performance.append(h)


# cv2.drawContours(img, ctrs, -1, color=(0, 255, 90),thickness=2)

# cv2.rectangle(img, (10,10), (300, 300), (255, 0, 123), 2)

# crop = img[10:300, 10:300]

# cv2.imshow('1', crop)
# cv2.imshow('2', img)
# cv2.waitKey(0)

# objects = ('1', '2', '3', '4', '5', '6', '7', '8')
# y_pos = np.arange(len(ctrs))
# # # performance = [10,8,6,4,2,1]

# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)


# plt.show()