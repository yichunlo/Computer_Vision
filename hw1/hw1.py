import cv2
import numpy as np

def task1_upside_down(img):
	ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
	for i in range(img.shape[0]):
		ret[i] = img[-i]
	return ret

def task2_right_side_left(img):
	ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
	for i in range(img.shape[1]):
		ret[:, i] = img[:, -i]
	return ret

def task3_diagonally_flip(img):
	ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			ret[i, j] = img[j, i]
	return ret
	
if __name__ == "__main__":
	img = cv2.imread('lena.bmp')
# task(a)
	task1 = task1_upside_down(img)
	cv2.imwrite('upside_down.png', task1)
# task(b)
	task2 = task2_right_side_left(img)
	cv2.imwrite('right_side_left.png', task2)
# task(c)
	task3 = task3_diagonally_flip(img)
	cv2.imwrite('diagonally_flip.png', task3)

