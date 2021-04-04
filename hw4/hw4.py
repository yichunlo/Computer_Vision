import cv2
import numpy as np
import matplotlib.pyplot as plt

ker = [[-2, -1], [-2, 0], [-2, 1], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [2, -1], [2, 0], [2, 1]]

def threshold(img):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i, j] < 128:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	return ret

def dilation(img, ker):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i, j] == 255:
				for idx in ker:
					if 0 <= i + idx[0] < img.shape[0] and 0 <= j + idx[1] < img.shape[1]:
						ret[i + idx[0], j + idx[1]] = 255
	return ret


def erosion(img, ker):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			flag = 1
			for idx in ker:
				if not (0 <= i + idx[0] < img.shape[0] and 0 <= j + idx[1] < img.shape[1]) \
					or img[i + idx[0], j + idx[1]] != 255:
					flag = 0
					break
			if flag:
				ret[i, j] = 255
	return ret

def opening(img, ker):
	ret = erosion(img, ker)
	ret = dilation(img, ker)
	return ret

def closing(img, ker):
	ret = dilation(img, ker)
	ret = erosion(img, ker)
	return ret

def reverse(img):
	ret = np.zeros((img.shape), np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i, j] == 0:
				ret[i, j] = 255
			else:
				ret[i, j] = 0
	return ret

def intersec(img1, img2):
	ret = np.zeros((img1.shape), np.int)
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			if img1[i, j] == 255 and img2[i, j] == 255:
				ret[i, j] = 255
	return ret

def hit_and_miss(img):
	# L shaped kernel
	L1 = [[0, 0], [0, -1], [1, 0]]
	L2 = [[0, 1], [-1, 0], [-1, 1]]

	ret = np.zeros((img.shape), np.int)
	i1 = reverse(img)
	i2 = erosion(img, L1)
	i3 = erosion(i1, L2)
	ret = intersec(i2, i3)
	return ret

if __name__ == '__main__':

	img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	GrayImage = threshold(img)
	#print(GrayImage)
	img = dilation(GrayImage, ker)
	#print(img)
	cv2.imwrite("dilation.bmp", img)
	cv2.imwrite("erosion.bmp", erosion(GrayImage, ker))
	cv2.imwrite("opening.bmp", opening(GrayImage, ker))
	cv2.imwrite("closing.bmp", closing(GrayImage, ker))
	cv2.imwrite("hit_and_miss.bmp", hit_and_miss(GrayImage))









