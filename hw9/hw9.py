import numpy as np
import cv2
import sys
import math

def extension(img):
    img_ext = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    img_ext[1:-1, 1:-1] = img[:, :]
    img_ext[0, 1:-1] = img[0, :]
    img_ext[-1, 1:-1] = img[-1, :]
    img_ext[1:-1, 0] = img[:, 0]
    img_ext[1:-1, -1] = img[:, -1]
    img_ext[0, 0] = img[0, 0]
    img_ext[0, -1] = img[0, -1]
    img_ext[-1, 0] = img[-1, 0]
    img_ext[-1, -1] = img[-1, -1]
    return img_ext

def robert(img, threshold):
	tmp = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			tmp[i, j] = img[i, j]
	for i in range(img.shape[0]):
		tmp[i, img.shape[1]] = img[i, img.shape[1] - 1]
	for j in range(img.shape[1]):
		tmp[img.shape[0], j] = img[img.shape[0] - 1, j]

	ret = np.zeros(img.shape, np.int);
	
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			ret[i, j] = 0
			r1 = tmp[i + 1, j + 1] - tmp[i,j]
			r2 = tmp[i + 1, j] - tmp[i, j + 1]
			gradient = (r1**2) + (r2**2)
			if gradient < (threshold**2):
				ret[i, j] = 255
	return ret

def prewitt(img, threshold):
	ret = np.zeros(img.shape, np.int)
	tmp = extension(img)
	for i in range(1, img.shape[0] + 1):
		for j in range(1, img.shape[1] + 1):
			ret[i - 1, j - 1] = 0
			p1 = tmp[i + 1, j - 1] + tmp[i + 1, j] + tmp[i + 1, j + 1] - tmp[i - 1, j - 1] - tmp[i - 1, j] - tmp[i - 1, j + 1]
			p2 = tmp[i - 1, j + 1] + tmp[i, j + 1] + tmp[i + 1, j + 1] - tmp[i - 1, j - 1] - tmp[i, j - 1] - tmp[i + 1, j - 1]
			gradient = (p1**2) + (p2**2)
			if gradient < (threshold**2):
				ret[i - 1, j - 1] = 255
	return ret

def sobel(img, threshold):
	ret = np.zeros(img.shape, np.int)
	tmp = extension(img)
	for i in range(1, img.shape[0] + 1):
		for j in range(1, img.shape[1] + 1):
			ret[i - 1, j - 1] = 255
			gi = tmp[i + 1, j - 1] + tmp[i + 1, j] * 2 + tmp[i + 1, j + 1] - tmp[i - 1, j - 1] - tmp[i - 1, j] * 2 - tmp[i - 1, j + 1]
			gj = tmp[i - 1, j + 1] + tmp[i, j + 1] * 2 + tmp[i + 1, j + 1] - tmp[i - 1, j - 1] - tmp[i, j - 1] * 2 - tmp[i + 1, j - 1]
			gradient = (gi**2) + (gj**2)
			if gradient >= (threshold**2):
				ret[i - 1, j - 1] = 0
	return ret

def frei_chen(img, threshold):
	ret = np.zeros(img.shape, np.int)
	tmp = extension(img)
	for i in range(1, img.shape[0] + 1):
		for j in range(1, img.shape[1] + 1):
			ret[i - 1, j - 1] = 0
			f1 = tmp[i + 1, j - 1] + tmp[i + 1, j] * math.sqrt(2) + tmp[i + 1, j + 1] - tmp[i - 1, j - 1] - tmp[i - 1, j] * math.sqrt(2) - tmp[i - 1, j + 1]
			f2 = tmp[i - 1, j + 1] + tmp[i, j + 1] * math.sqrt(2) + tmp[i + 1, j + 1] - tmp[i - 1, j - 1] - tmp[i, j - 1] * math.sqrt(2) - tmp[i + 1, j - 1]
			gradient = (f1**2) + (f2**2)
			if gradient < (threshold**2):
				ret[i - 1, j - 1] = 255
	return ret

def kirsch(img, threshold):
	ret = np.zeros(img.shape, np.int)
	tmp = extension(img)
	k = [-3, -3, 5, 5, 5, -3, -3, -3]
	for i in range(1, img.shape[0] + 1):
		for j in range(1, img.shape[1] + 1):
			ret[i - 1, j - 1] = 0
			gradient = 0
			for x in range(8):
				gx = tmp[i - 1, j - 1] * k[x] + tmp[i - 1, j] * k[(x+1)%8] + tmp[i - 1, j + 1] * k[(x+2)%8] + tmp[i, j + 1] * k[(x+3)%8] + tmp[i + 1, j + 1] * k[(x+4)%8] + tmp[i + 1, j] * k[(x+5)%8] + tmp[i + 1, j - 1] * k[(x+6)%8] +tmp[i, j - 1] * k[(x+7)%8]
				if gx > gradient:
					gradient = gx

			if gradient < threshold:
				ret[i - 1, j - 1] = 255
	return ret

def robinson(img, threshold):
	ret = np.zeros(img.shape, np.int)
	tmp = extension(img)
	k = [-1, 0, 1, 2, 1, 0, -1, -2]
	for i in range(1, img.shape[0] + 1):
		for j in range(1, img.shape[1] + 1):
			ret[i - 1, j - 1] = 0
			gradient = 0
			for z in range(8):
				r = tmp[i - 1, j - 1] * k[z] + tmp[i - 1, j] * k[(z+1)%8] + tmp[i - 1, j + 1] * k[(z+2)%8] + tmp[i, j + 1] * k[(z+3)%8] + tmp[i + 1, j + 1] * k[(z+4)%8] + tmp[i + 1, j] * k[(z+5)%8] + tmp[i + 1, j - 1] * k[(z+6)%8] +tmp[i, j - 1] * k[(z+7)%8]
				if r > gradient:
					gradient = r

			if gradient < threshold:
				ret[i - 1, j - 1] = 255
	return ret

def nevatia(img, threshold):
	ret = np.zeros(img.shape, np.int)
	tmp = extension(extension(img))
	kernel = [ [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2],
				[-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
				[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
				[-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
				[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2] ]
	k0 = [ 100, 100, 0, -100, -100, 100, 100, 0, -100, -100, 100, 100, 0,
			-100, -100, 100, 100, 0, -100, -100, 100, 100, 0, -100, -100]
	k1 = [ 100, 100, 100, 100, 100, 100, 100, 100, 78, -32, 100, 92, 0, -92,
			-100, 32, -78, -100, -100, -100, -100, -100, -100, -100, -100]
	k2 = [-100, -100, -100, -100, -100, 32, -78, -100, -100, -100, 100, 92,
			0, -92, -100, 100, 100, 100, 78, -32, 100, 100, 100, 100, 100]
	k3 = [ 100, 100, 100, 32, -100, 100, 100, 92, -78, -100, 100, 100, 0,
			-100, -100, 100, 78, -92, -100, -100, 100, -32, -100, -100, -100]
	k4 = [ -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0,
			0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
	k5 = [ 100, -32, -100, -100, -100, 100, 78, -92, -100, -100, 100, 100,
			0, -100, -100, 100, 100, 92, -78, -100, 100, 100, 100, 32, -100]

	for i in range(2, img.shape[0] + 2):
		for j in range(2, img.shape[1] + 2):
			ret[i - 2, j - 2] = 0
			cnt = 0
			g0 = g1 = g2 = g3 = g4 = g5 = 0
			for z in kernel:
				g0 += tmp[i + z[0], j + z[1]] * k0[cnt]
				g1 += tmp[i + z[0], j + z[1]] * k1[cnt]
				g2 += tmp[i + z[0], j + z[1]] * k2[cnt]
				g3 += tmp[i + z[0], j + z[1]] * k3[cnt]
				g4 += tmp[i + z[0], j + z[1]] * k4[cnt]
				g5 += tmp[i + z[0], j + z[1]] * k5[cnt]
				cnt += 1

			gradient = max(g0, g1, g2, g3, g4, g5)
			if gradient < threshold:
				ret[i - 2, j - 2] = 255
	return ret

img_gray = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imwrite("roberts12.bmp", robert(img_gray, 12))
cv2.imwrite("prewitt24.bmp", prewitt(img_gray, 24))
cv2.imwrite("sobel38.bmp", sobel(img_gray, 38))
cv2.imwrite("frei_chen30.bmp", frei_chen(img_gray, 30))
cv2.imwrite("kirsch135.bmp", kirsch(img_gray, 135))
cv2.imwrite("robinson43.bmp", robinson(img_gray, 43))
cv2.imwrite("nevatia12500.bmp", nevatia(img_gray, 12500))

