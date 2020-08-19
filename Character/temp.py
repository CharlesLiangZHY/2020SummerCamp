import cv2 as cv
import numpy as np

img = cv.imread('b.jpg')
width = img.shape[1]

def padding(img):
	h,w = img.shape[0:2]
	new_img = np.zeros([512, 512, 3], dtype=np.uint8)
	new_img[:,:,:] = 255
	new_img[(512-h)//2 : 512-(512-h)//2 , (512-w)//2: 512-(512-w)//2, :] = img
	return new_img

def resize(img,n):
	h,w = img.shape[0:2]
	return cv.resize(img,(int(w/n), int(h/n)))


img = resize(img, width//512 + 1)
img = padding(img)

cv.imshow('test', img)
cv.waitKey()