import cv2
import numpy as np
import matplotlib.pyplot as plt

ker = [[-2, -1], [-2, 0], [-2, 1], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [2, -1], [2, 0], [2, 1]]

def dilation(img, ker):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			maxv = 0
			for idx in ker:
				if 0 <= i - idx[0] < img.shape[0] and 0 <= j - idx[1] < img.shape[1]:
					if img[i - idx[0], j - idx[1]] > maxv:
						maxv = img[i - idx[0], j - idx[1]]
			ret[i - idx[0], j - idx[1]] = maxv
	return ret


def erosion(img, ker):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			minv = 255
			for idx in ker:
				if 0 <= i + idx[0] < img.shape[0] and 0 <= j + idx[1] < img.shape[1]:
					if img[i + idx[0], j + idx[1]] < minv:
						minv = img[i + idx[0], j + idx[1]]
			ret[i, j] = minv
	return ret

def opening(img, ker):
	ret = erosion(img, ker)
	ret = dilation(ret, ker)
	return ret

def closing(img, ker):
	ret = dilation(img, ker)
	ret = erosion(ret, ker)
	return ret


if __name__ == '__main__':

	img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	#print(img)
	cv2.imwrite("dilation.bmp", dilation(img, ker))
	cv2.imwrite("erosion.bmp", erosion(img, ker))
	cv2.imwrite("opening.bmp", opening(img, ker))
	cv2.imwrite("closing.bmp", closing(img, ker))









