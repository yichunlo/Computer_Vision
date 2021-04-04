import numpy as np
import cv2
import sys
np.set_printoptions(threshold = sys.maxsize)

def func(img):
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
	ans = np.zeros((64, 64), np.int)
	for i in range(1, 65):
		for j in range(1, 65):
			if img[j, j] ==  255:
				flag = 0
				count = 0
				cnt = 0
				for k in range(9):
					if img[i + check[k % 8][0], j + check[k % 8][1]] == 255:
						cnt += 1
						if flag == 1 and k == 8:
							count -= 1
						if flag == 0 and k % 2 == 0 and k < 8:
							count += 1
							flag = 1
					else:
						if flag == 1:
							flag = 0
				if cnt == 9:
					ans[i - 1, j - 1] = 5
				else:
					ans[i - 1, j - 1] = count

	f = open("yokoi.txt", "w")
	for i in range(64):
		for j in range(64):
			if ans[i, j] == 0:
				f.write(" ")
			else:
				f.write(str(ans[i, j]))
			f.write(" ")
		f.write("\n")
img_gray = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imwrite("sample.bmp", func(img_gray))
yokoi(func(img_gray))
