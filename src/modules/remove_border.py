import numpy as np
import cv2
from deskewing import deskewing
file_path = r"C:\Users\16521\Music\work\d2d\D2D_2.png"

def border(image):
    # image = cv2.imread(image)
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = 255*(gray < 85).astype(np.uint8) # To invert the text to white
    # cv2.imshow("das", gray)
    # cv2.waitKey()
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2,2), dtype=np.uint8))
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    # x:left, x+w: right, y:top, y+h:bot
    rect = image[y-20:y+h+20, x-20:x+w+20] # Crop the image - note we do this on the original image
    return rect
def findBoxContours(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    list_area = []
    boxes = []
    
    for c in contours:
        if cv2.contourArea(c) > 50000:
            x, y, w, h = cv2.boundingRect(c)
            box = (x, y, x + w, y + h)
            list_area.append(cv2.contourArea(c))
            boxes.append(box)
    return boxes

img = cv2.imread(file_path)
# img = deskewing(img)
boxes = findBoxContours(img)
print(boxes)
for pt in boxes:
    cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (0, 90, 255), 3)
cv2.imwrite("temp.png", img)
