import cv2
import numpy as np
from deskewing import deskewing

def sort_contours(cnts):
    boundingBoxs = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxs) = zip(*sorted(zip(cnts, boundingBoxs), key=lambda b:b[1][0], reverse=False))
    return cnts

def box_extraction(image):

    # image = cv2.imread(img_for_box_extraction_path)
    # print(image.shape)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image
    img_bin = cv2.dilate(img_bin,kernel,iterations = 1) # khi sử dụng ảnh image_result_table.jpg
    #img_bin = cv2.erode(img_bin,kernel,iterations = 1)
    # cv2.imshow("Image_bin.jpg",img_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # cv2.imshow("verticle_lines.jpg",verticle_lines_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    # cv2.imshow("horizontal_lines.jpg",horizontal_lines_img)
    # cv2.waitKey()
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    # cv2.imshow("img_final_bin.jpg",img_final_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    # (contours, boundingBoxes) = sort_contours(contours)
    img_final = None
    for c in contours:
        # Returns the location and width,height for every contour
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h < 3000000 and w*h> 700000:

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        # if (w > 80 and h > 20) and w > 3*h:
        # idx += 1
        # cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0), 4)
            new_img = image[y:y+h, x:x+w]
            # cv2.imwrite("temp.png", new_img)
            return new_img
    # return img_final
    # img = cv2.resize(image, (800, 600))
    # cv2.imshow(cropped_dir_path+str(idx) + '.png', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./Temp/img_contour.jpg", img)
def table_extract(image):
    # image = cv2.resize(image, (800, 700))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image
    img_bin = cv2.dilate(img_bin,kernel,iterations = 1) # khi sử dụng ảnh image_result_table.jpg
    #img_bin = cv2.erode(img_bin,kernel,iterations = 1)
    # cv2.imshow("Image_bin.jpg",img_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # cv2.imshow("verticle_lines.jpg",verticle_lines_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    # cv2.imshow("horizontal_lines.jpg",horizontal_lines_img)
    # cv2.waitKey()
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    # print(img_final_bin.shape)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print(thresh.shape)
    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    # cv2.imshow("img_final_bin.jpg",img_final_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Sort all the contours by top to bottom.
    # contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1] * image.shape[1] )
    n_contours = sort_contours(contours)
    result = []
    for c in n_contours:
        # Returns the location and width,height for every contour
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h < image.shape[0]*image.shape[1] and w*h>2000:
        # if w*h < 3000000 and w*h> 700000:

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        # if (w > 80 and h > 20) and w > 3*h:
        # idx += 1
            # cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0), 2)
            new_img = image[y:y+h, x:x+w]
            # # print(new_img.shape)
            # result.append(new_img)
            result.append((x, y, w, h))
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
            # result.append(new_img)
    return result
    # cv2.imshow('.png', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()



# img_path = r"D:\Desktop\PBE\Images\HSCT.010.18.00000129#0008-2.png"
img_path = r"D:\Desktop\PBE\Images\96.signed_01-07.jpg"
image = cv2.imread(img_path)
# des_img = deskewing(image)
img = box_extraction(image)

# cv2.imshow('1', img)
# cv2.waitKey(0)
img = deskewing(img)
cv2.namedWindow('2', cv2.WINDOW_NORMAL)
cv2.imshow('2', img)
cv2.waitKey(0)
# cv2.destroyWindow('2')
# finals =  table_extract(img)
# # # i = 0
# for f in finals:
    # print(f)
#     cv2.imshow('1', f)
#     cv2.waitKey(0)
#     cv2.destroyWindow('1')
#     cv2.imshow('/crop'+str(i)+'.png', f)
#     i = i + 1

# box_extraction("image_result_table_2.jpg", "./Cropped/")
# box_extraction("41.jpg", "./Cropped/")