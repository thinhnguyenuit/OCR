import cv2
import numpy as np

'''
def square(img):
    """
    This function resize non square image to square one (height == width)
    :param img: input image as numpy array
    :return: numpy array
    """

    # image after making height equal to width
    squared_image = img

    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))

        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))

        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image
'''

#def char_segmentation(img_file_path):
img_file_path = r"D:\Desktop\HandReco\SimpleHTR\data\a01-000u-s00-01.png"
img = cv2.imread(img_file_path)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = img.shape[0]
width = img.shape[1]
area = height * width

scale1 = 0.01
scale2 = 0.1
area_condition1 = area * scale1
area_condition2 = area * scale2

# Otsu's thresholding
ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(imgray,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


new_contours = []

for i in range(len(contours)):
    if hierarchy[0][i][2] == -1 :
        new_contours.append(contours[i])

new_contours = sorted(new_contours,key=lambda b:b[0][0][0], reverse = False)
cropped = []
for cnt in contours:
    (x,y,w,h) = cv2.boundingRect(cnt)
    cv2.drawContours(img,cnt,-1,(0,0,255),2)
    #cv2.rectangle(img,(x,x+w),(y,y+h),(0,255,0),2)
    if (w * h > area_condition1 and w * h < area_condition2 and w/h > 0.1 or h/w > 0.1):
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        c = th2[y:y+h,x:x+w]
        c = np.array(c)
        c = cv2.bitwise_not(c)
        #c = square(c)
        #c = cv2.resize(c,(28,28), interpolation = cv2.INTER_AREA)
        cropped.append(c)

    #return cropped

cv2.imshow("1", img)
cv2.waitKey(0)

'''
#img_file_path = r"D:\Desktop\PBE\ChuVietTayPhieuDieuTra_Box.png"
img_file_path = r"D:\Desktop\HandReco\SimpleHTR\data\a01-000u-s00-01.png"
#img_file_path = r'D:\Desktop\PBE\img20190110_0002-1.jpg'
crop = char_segmentation(img_file_path)
for item in crop:
    cv2.imshow("1", item)
    cv2.waitKey(0)
'''