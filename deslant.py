import numpy as np
import cv2



def deskew(img):
    h, w = img.shape
    SZ = 100
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 1]])
    img = cv2.warpAffine(img,M,(w, h),flags=affine_flags)
    return img

# img_path = r'D:\OCR\Dataset\data_subset\data_subset\a01-011u-s01-01.png'
# image = cv2.imread(img_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = deskew(gray)
# cv2.namedWindow("1", cv2.WINDOW_NORMAL)
# cv2.imshow("1" ,img)
# cv2.waitKey(0)