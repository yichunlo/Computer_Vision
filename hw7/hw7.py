import numpy as np
import cv2
import sys
np.set_printoptions(threshold = sys.maxsize)

def ds(img):
	ret = np.zeros((66, 66), np.int)
	for i in range(64):
		for j in range(64):
			if img[i * 8, j * 8] >= 128:
				ret[i + 1, j + 1] = 255
			else:
				ret[i + 1, j + 1] = 0
	return ret

def yokoi(img):
	check = [[0, 1], [1, 1], [1, 0], [1, -1],
			[0, -1], [-1, -1], [-1, 0], [-1, 1]]
	ret = np.zeros((66, 66), np.int)
	for i in range(1, 65):
		for j in range(1, 65):
			if img[i, j] ==  255:
				flag = 0
				num = 0
				cnt = 0
				for k in range(9):
					if img[i + check[k % 8][0], j + check[k % 8][1]] == 255:
						cnt += 1
						if flag == 1 and k == 8:
							num -= 1
						if flag == 0 and k % 2 == 0 and k < 8:
							num += 1
							flag = 1
					else:
						if flag == 1:
							flag = 0
				if cnt == 9:
					ret[i, j] = 5
				else:
					ret[i, j] = num
	return ret

def get_h(arr):
	ret = np.zeros((66, 66))
	for i in range(1, 65):
		for j in range(1, 65):
			if arr[i, j] == 1:
				if (i >= 2 and arr[i - 1, j] == 1) or (i < 64 and arr[i + 1, j] == 1) \
				or (j >= 2 and arr[i, j - 1] == 1) or (j < 64 and arr[i, j + 1] == 1):
					ret[i, j] = 100
			elif arr[i, j] != 0:
				ret[i, j] = 99
	return ret

def thining(img, img2):
	check = [[0, 1], [1, 1], [1, 0], [1, -1],
			[0, -1], [-1, -1], [-1, 0], [-1, 1]]
	for i in range(1, 65):
		for j in range(1, 65):
			if img[i, j] ==  255:
				flag = 0
				num = 0
				cnt = 0
				for k in range(9):
					if img[i + check[k % 8][0], j + check[k % 8][1]] == 255:
						cnt += 1
						if flag == 1 and k == 8:
							num -= 1
						if flag == 0 and k % 2 == 0 and k < 8:
							num += 1
							flag = 1
					else:
						if flag == 1:
							flag = 0
				if num == 1 and img2[i, j] == 100:
					img[i, j] = 0
	return img

img_gray = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
img = ds(img_gray)
for i in range(7):
	img2 = yokoi(img)
	tmp_img = get_h(img2)
	img = thining(img, tmp_img)

ans = np.zeros((64, 64))
for i in range(64):
	for j in range(64):
		ans[i, j] = img[i + 1, j + 1]

cv2.imwrite("thining.bmp", ans)