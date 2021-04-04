import numpy as np
import matplotlib.pyplot as plt
import cv2

max_ = 100000000

def task1_threshold(img):
	ret = np.zeros((img.shape), np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i, j] < 128:
				ret[i, j] = 0
			else:
				ret[i, j] = 255
	return ret
	
def task2_histogram(img):
	ret = np.zeros(256)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			ret[img[i, j]] += 1
	plt.bar(range(len(ret)), ret, width = 1.5)
#plt.show()
	
def task3_connected(img1, img2):
	binary = task1_threshold(img1)
	visited = np.zeros((img1.shape))
	ret = img2.copy()
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			r_sum, c_sum = 0, 0
			left, right, up, down, cnt = max_, -max_, max_, -max_, 0
			stk = [(i, j)]
			while stk:
				r, c = stk.pop()
				if 0 <= r < 512 and 0 <= c < 512 and visited[r, c] == 0 and binary[r, c] != 0:
					cnt += 1
					r_sum += r
					c_sum += c
					visited[r, c] = 1
					left  = min(left,  c)
					right = max(right, c)
					up    = min(up,    r)
					down  = max(down,  r)
					stk.extend([(r, c+1), (r, c-1), (r+1, c), (r-1, c)])
			if cnt >= 500:
				cv2.rectangle(ret, (left, down), (right, up), (255, 0, 0), 3)
				cv2.circle(ret, (int(c_sum / cnt), int(r_sum / cnt)), 4, (0, 0, 255), -1)
	return ret


if __name__ == "__main__":
	img1 = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	img2 = cv2.imread('lena.bmp')
	ret1 = task1_threshold(img1)
	cv2.imwrite('threshold.bmp', ret1)
	task2_histogram(img1)
	ret2 = task3_connected(img1, img2)
	cv2.imwrite('connected.bmp', ret2)

