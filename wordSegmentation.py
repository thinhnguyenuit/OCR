import math
import cv2
import numpy as np

def wordSegmentation(img, kernelSize, sigma, theta, minArea):
	# apply filter kernel
	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	cv2.imshow("1", imgFiltered)
	cv2.waitKey(0)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	imgThres = 255 - imgThres

	# find connected components. OpenCV: return type differs between OpenCV2 and 3
	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# append components to result
	res = []
	for c in components:
		# skip small word candidates
		if cv2.contourArea(c) < minArea:
			continue
		# append bounding box and image of word to result list
		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	# return list of words, sorted by x-coordinate
	return sorted(res, key=lambda entry:entry[0][0])



def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2

	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta

	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize

			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel


def wordSegment(img):
    kernel_size = 33
    sigma = 11
    theta = 11
    minRect = 900
    blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = wordSegmentation(blur, kernel_size, sigma, theta, minRect)
    return res


img_file_path = r"D:\Desktop\HandReco\SimpleHTR\data\a01-000u-s00-01.png"
#img_file_path = r"vd1.jpg"
img = cv2.imread(img_file_path)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = wordSegmentation(imgray, 31, 7, 7, 150)
for (j, w) in enumerate(res):
    (wordBox, wordImg) = w
    (x, y, w, h) = wordBox
    #cv2.imwrite('../out/%s/%d.png'%(f, j), wordImg)
    cv2.rectangle(img,(x,y),(x+w,y+h),0,1)

cv2.namedWindow("1", cv2.WINDOW_GUI_EXPANDED)
cv2.imshow("1", img)
cv2.waitKey(0)
