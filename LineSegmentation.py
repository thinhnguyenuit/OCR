import cv2
import numpy as np
#import image
#image = cv2.imread('vd3.jpg')
#mage = cv2.resize(image, None, fx=1.5, fy= 1.5, interpolation=cv2.INTER_CUBIC)
#cv2.imshow('orig',image)
#cv2.waitKey(0)
def LineSegment(image):
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("3", thresh)
    cv2.waitKey(0)

    #dilation
    kernel = np.ones((3,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(ctrs) == 0:
        ret,thresh = cv2.threshold(gray,90,255,cv2.THRESH_BINARY_INV)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    boundingbox = [cv2.boundingRect(c) for c in ctrs]
    (sorted_ctrs, boundingbox) = zip(*sorted(zip(ctrs,boundingbox), key=lambda b:b[1][1], reverse = False))

    res = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        if cv2.contourArea(ctr) > 1500:
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = image[y:y+h, x:x+w]
            res.append((cv2.boundingRect(ctr),roi))
            # show ROI
            # cv2.imshow('segment no:'+str(i),roi)
            #cv2.rectangle(image,(x,y-10),( x + w, y + h ),(90,0,255),2)
            # cv2.waitKey(0)

    return res
#res1 =  zip(*sorted(res, key=lambda entry:entry[1][0], reverse = True))

# for r in res:
#     cv2.imshow("1", r[1])
#     cv2.waitKey(0)
# cv2.namedWindow('marked', cv2.WINDOW_NORMAL)
# cv2.imshow('marked',image)
# cv2.waitKey(0)