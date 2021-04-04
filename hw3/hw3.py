import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_histogram(img):
	res = np.zeros(256)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			res[int(img[i][j])] += 1
	plt.bar(range(len(res)), res, width = 2)
	plt.show()

def equalize(img):
	ret = np.zeros(img.shape)
	cnt = np.zeros(256)
	equ = np.zeros(256)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			cnt[int(img[i, j])] += 1
	cdf = 0
	for i in range(256):
		cdf += cnt[i]
		equ[i] = (255 * cdf) // (512 ** 2)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			ret[i, j] = equ[int(img[i, j])]
	return ret

if __name__ == "__main__":
	img1 = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	make_histogram(img1)
	img2 = img1 / 3
	cv2.imwrite('devided_by_3.bmp', img2)
	make_histogram(img2)
	img3 = equalize(img2)
	make_histogram(img3)
	cv2.imwrite('equalized.bmp', img3)

	

