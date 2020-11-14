import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import time

img = cv2.imread('test.jpg')
# SIZE = ( int(3096/3), int(4128/3))
SIZE = ( int(3096), int(4128))

# img = cv2.resize(img, SIZE)

cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, bw_img) = cv2.threshold(img_rgb, 140, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(bw_img,140,255)
lines = cv2.HoughLines(edges,1,np.pi/180,200)

cv2.imshow('image', bw_img)

xs=[]

for item in lines:
    for rho,theta in item:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        xs.append(max(x1,x2))
        # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

print(img_rgb.shape)
img_rgb = img_rgb[0:SIZE[1], xs[0]:xs[1]]
img = img[0:SIZE[1], xs[0]:xs[1]]
edges = img_rgb[0:SIZE[1], xs[0]:xs[1]]
# string = pytesseract.image_to_string(edges)
# print(string)
# img= cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('image', img)

# OTSU THRESHOLD
ret,thresh1 = cv2.threshold(img_rgb, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
cv2.imshow('thresh1', thresh1)

# DILATION
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 12))
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
dilation_pic= cv2.rotate(dilation, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('dilation', dilation_pic)

# FINDING CONTOURS
# (4128, 3096)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
'''
im2 = img_rgb.copy()
for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print((x, y), (x + w, y + h))
        # cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 2000000000055, 0), 2)
        mtxt = im2[ y:y+h, x:x + w]
        (thresh, mtxt_bw) = cv2.threshold(mtxt, 150, 255, cv2.THRESH_BINARY)
        string = pytesseract.image_to_string(mtxt_bw)
        print(string)
        cv2.imshow('final', mtxt_bw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
im2= cv2.rotate(im2, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('final', im2)
'''


# string = pytesseract.image_to_string(img_rgb)
# print(string)
cv2.waitKey(0)
cv2.destroyAllWindows()