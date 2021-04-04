import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

kernel = [  [0, 0], [0, 1], [0, 2], [0, -1], [0, -2],
			[1, 0], [1, 1], [1, 2], [1, -1], [1, -2],
			[-1, 0], [-1, 1], [-1, 2], [-1, -1], [-1, -2],
			[2, -1], [2, 0], [2, 1],
			[-2, -1], [-2, 0], [-2, 1] ]

def gaussion(img, amp):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			ret[i, j] = img[i, j] + amp * np.random.normal(0, 1)
	return ret

def saltAndPepper(img, threshold):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			uniform = np.random.uniform(0, 1)
			if uniform > 1 - threshold:
				ret[i, j] = 255
			elif uniform < threshold:
				ret[i, j] = 0
			else:
				ret[i, j] = img[i, j]
	return ret

def box(img, siz):
	ret = np.zeros((img.shape))
	tmp = np.zeros((img.shape[0] + siz -1, img.shape[1] + siz - 1))
	half = int((siz - 1) / 2)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			tmp[i + half, j + half] = img[i, j]

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			count = 0
			l = int(-half)
			r = int(half + 1)
			for a in range(l, r):
				for b in range(l, r):
					count += tmp[i + half + a, j + half + b]
			ret[i, j] = round(count / (siz * siz))
	return ret

def median(img, siz):
	ret = np.zeros((img.shape))
	tmp = np.zeros((img.shape[0] + siz -1, img.shape[1] + siz - 1))
	half = int((siz - 1) / 2)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			tmp[i + half, j + half] = img[i, j]

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			total = []
			for a in range(int(-half), int(half + 1)):
				for b in range(int(-half), int(half +1)):
					total.append(tmp[i + half + a, j + half + b])
			ret[i, j] = np.median(total)
	return ret

def dilation(img):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			maxn = 0
			for idx in kernel:
				if img.shape[0] > i - idx[0] >= 0 and img.shape[1] > j - idx[1] >= 0:
					if img[i - idx[0], j - idx[1]]  > maxn:
						maxn = img[i - idx[0], j - idx[1]]
			ret[i, j] = maxn
	return ret

def erosion(img):
	ret = np.zeros((img.shape))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			mini = 255
			for idx in kernel:
				if img.shape[0] > i + idx[0] >= 0 and img.shape[1] > j + idx[1] >= 0:
					if img[i + idx[0], j + idx[1]] < mini:
						mini = img[i + idx[0], j + idx[1]]
			ret[i, j] = mini
	
	return ret

def opening(img):
	ret = erosion(img)
	ret = dilation(ret)
	return ret

def closing(img):
	ret = dilation(img)
	ret = erosion(ret)
	return ret

def snr(img, noise):
	vs = u = vn = un = 0
	ret = 0
	siz = img.shape[0] * img.shape[1]
	for i in range (img.shape[0]):
		for j in range(img.shape[1]):
			u += img[i, j]
	u /= siz

	for i in range (img.shape[0]):
		for j in range(img.shape[1]):
			vs += (img[i, j] - u)**2
	vs /= siz

	for i in range( noise.shape[0]):
		for j in range(noise.shape[1]):
			un += (noise[i, j] - img[i, j])
	un /= siz
	
	for i in range( noise.shape[0]):
		for j in range(noise.shape[1]):
			vn += (noise[i, j] - img[i, j] - un)**2
	vn /=  siz

	ret_snr = 20 * math.log( math.sqrt(vs) / math.sqrt(vn), 10)

	return ret_snr

img_gray = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
g10 = gaussion(img_gray, 10)
g30 = gaussion(img_gray, 30)
sap005 = saltAndPepper(img_gray, 0.05)
sap01 = saltAndPepper(img_gray, 0.1)
box_g10_3 = box(g10, 3)
box_g30_3 = box(g30, 3)
box_sap005_3 = box(sap005, 3)
box_sap01_3 = box(sap01, 3)
box_g10_5 = box(g10, 5)
box_g30_5 = box(g30, 5)
box_sap005_5 = box(sap005, 5)
box_sap01_5 = box(sap01, 5)
median_g10_3 = median(g10, 3)
median_g30_3 = median(g30, 3)
median_sap005_3 = median(sap005, 3)
median_sap01_3 = median(sap01, 3)
median_g10_5 = median(g10, 5)
median_g30_5 = median(g30, 5)
median_sap005_5 = median(sap005, 5)
median_sap01_5 = median(sap01, 5)
oc_g10 = closing(opening(g10))
co_g10 = opening(closing(g10))
oc_g30 = closing(opening(g30))
co_g30 = opening(closing(g30))
oc_sap005 = closing(opening(sap005))
co_sap005 = opening(closing(sap005))
oc_sap01 = closing(opening(sap01))
co_sap01 = opening(closing(sap01))

cv2.imwrite("gaussian_10.bmp", g10)
cv2.imwrite("guassion_30.bmp", g30)
cv2.imwrite("salt_and_pepper005.bmp", sap005)
cv2.imwrite("salt_and_pepperp01.bmp", sap01) 
cv2.imwrite("box_g10_3x3.bmp", box_g10_3)
cv2.imwrite("box_g30_3x3.bmp", box_g30_3)
cv2.imwrite("box_sap005_3x3.bmp", box_sap005_3)
cv2.imwrite("box_sap01_3x3.bmp", box_sap01_3)
cv2.imwrite("box_g10_5x5.bmp", box_g10_5)
cv2.imwrite("box_g30_5x5.bmp", box_g30_5)
cv2.imwrite("box_sap005_5x5.bmp", box_sap005_5)
cv2.imwrite("box_sap01_5x5.bmp", box_sap01_5)
cv2.imwrite("median_g10_3x3.bmp", median_g10_3)
cv2.imwrite("median_g30_3x3.bmp", median_g30_3)
cv2.imwrite("median_sap005_3x3.bmp", median_sap005_3)
cv2.imwrite("median_sap01_3x3.bmp", median_sap01_3)
cv2.imwrite("median_g10_5x5.bmp", median_g10_5)
cv2.imwrite("median_g30_5x5.bmp", median_g30_5)
cv2.imwrite("median_sap005_5x5.bmp", median_sap005_5)
cv2.imwrite("median_sap01_5x5.bmp", median_sap01_5)
cv2.imwrite("co_g10.bmp", co_g10)
cv2.imwrite("oc_g10.bmp", oc_g10)
cv2.imwrite("co_g30.bmp", co_g30)
cv2.imwrite("oc_g30.bmp", oc_g30)
cv2.imwrite("co_sap005.bmp", co_sap005)
cv2.imwrite("oc_sap005.bmp", oc_sap005)
cv2.imwrite("co_sap01.bmp", co_sap01)
cv2.imwrite("oc_sap01.bmp", oc_sap01)

print("============================================\n")
print("guassion_10:  ",snr(img_gray, g10))
print("boxing_10_3X3:",snr(img_gray, box_g10_3))
print("boxing_10_5x5:",snr(img_gray, box_g10_5))
print("median_10_3x3:",snr(img_gray, median_g10_3))
print("median_10_5x5:",snr(img_gray, median_g10_5))
print("open_close_10:",snr(img_gray, oc_g10))
print("close_open_10:",snr(img_gray, co_g10))
print("============================================\n")

print("guassion_30:  ",snr(img_gray, g30))
print("boxing_30_3x3:",snr(img_gray, box_g30_3))
print("boxing_30_5x5:",snr(img_gray, box_g30_5))
print("median_30_3x3:",snr(img_gray, median_g30_3))
print("median_30_5x5:",snr(img_gray, median_g30_5))
print("open_close_30:",snr(img_gray, oc_g30))
print("close_open_30:",snr(img_gray, co_g30))
print("============================================\n")

print("salt_pepper_5:",snr(img_gray, sap005))
print("box_sap_5_3x3:",snr(img_gray, box_sap005_3))
print("box_sap_5_5x5:",snr(img_gray, box_sap005_5))
print("median_sap5_3:",snr(img_gray, median_sap005_3))
print("median_sap5_5:",snr(img_gray, median_sap005_5))
print("open_clo_sap5:",snr(img_gray, oc_sap005))
print("clo_open_sap5:",snr(img_gray, co_sap005))
print("============================================\n")

print("salt_pepper_1:",snr(img_gray, sap01))
print("box_sap_1_3x3:",snr(img_gray, box_sap01_3))
print("box_sap_1_5x5:",snr(img_gray, box_sap01_5))
print("median_sap1_3:",snr(img_gray, median_sap01_3))
print("median_sap1_5:",snr(img_gray, median_sap01_5))
print("open_clo_sap1:",snr(img_gray, oc_sap01))
print("clo_open_sap1:",snr(img_gray, co_sap01))
print("============================================\n")

