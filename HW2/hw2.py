import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

## Problem 1-a

### first order
img1 = cv2.imread('sample1.jpg', cv2.IMREAD_GRAYSCALE)
img1_height = img1.shape[0]
img1_width = img1.shape[1]

def first_order_edge(k, src, height, width, threshold):
	dst = np.zeros((height, width), dtype = 'uint8')
	row_filter = np.array([[-1, 0, 1], [-1 * k, 0, k], [-1, 0, 1]])
	col_filter = np.array([[-1, -1 * k, -1], [0, 0, 0], [1, k, 1]])
	for i in range (height):
		for j in range (width):
			row_magnitude = 0
			col_magnitude = 0
			for m in range(3):
				for n in range(3):
					if (i + m - 1 >= 0 and i + m - 1 < height and j + n - 1 >= 0 and j + n - 1 < width):
						row_magnitude += int(src[i + m - 1][j + n - 1]) * row_filter[m][n]
						col_magnitude += int(src[i + m - 1][j + n - 1]) * col_filter[m][n]
					else:
						row_magnitude += int(src[i][j]) * row_filter[m][n]
						col_magnitude += int(src[i][j]) * col_filter[m][n]
			magnitude = math.sqrt(math.pow(row_magnitude, 2) + math.pow(col_magnitude, 2))
			if (magnitude > threshold * (k + 2)):
				dst[i][j] = 255
			else:
				dst[i][j] = 0
	return dst
'''
img2 = first_order_edge(1, img1, img1_height, img1_width, 30)
cv2.imwrite("result1_prewitt_30.jpg", img2)
img3 = first_order_edge(1, img1, img1_height, img1_width, 40)
cv2.imwrite("result1_prewitt_40.jpg", img3)
img4 = first_order_edge(1, img1, img1_height, img1_width, 50)
cv2.imwrite("result1_prewitt_50.jpg", img4)
img5 = first_order_edge(2, img1, img1_height, img1_width, 40)
cv2.imwrite("result1_sobel_40.jpg", img5)
img6 = first_order_edge(2, img1, img1_height, img1_width, 50)
cv2.imwrite("result1_sobel_50.jpg", img6)
img7 = first_order_edge(2, img1, img1_height, img1_width, 60)
cv2.imwrite("result1_sobel_60.jpg", img7)
'''
img2 = first_order_edge(1, img1, img1_height, img1_width, 30)
cv2.imwrite("result1.jpg", img2)

### second order
LoG_kernel_9 = np.array([[0, 1, 1, 2, 2, 2, 1, 1, 0],\
						[1, 2, 4, 5, 5, 5, 4, 2, 1],\
						[1, 4, 5, 3, 0, 3, 5, 4, 1],\
						[2, 5, 3, -12, -24, -12, 3, 5, 2],\
						[2, 5, 0, -24, -40, -24, 0, 5, 2],\
						[2, 5, 3, -12, -24, -12, 3, 5, 2],\
						[1, 4, 5, 3, 0, 3, 5, 4, 1],\
						[1, 2, 4, 5, 5, 5, 4, 2, 1],\
						[0, 1, 1, 2, 2, 2, 1, 1, 0]])
LoG_kernel_11 = np.array([	[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],\
						[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
						[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
						[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
						[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
						[-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],\
						[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
						[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
						[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
						[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
						[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])

def LaplacianofGaussian(src, kernel, kernel_size, height, width, threshold):
	log = np.zeros((height, width), dtype = 'int8')
	border_size = round(kernel_size / 2)
	for i in range(height):
		for j in range(width):
			val = 0
			for m in range(kernel_size):
				for n in range(kernel_size):
					i_tmp = i + m - border_size
					j_tmp = j + n - border_size
					if (i_tmp < 0): 
						i_tmp = 0
					elif (i_tmp >= height):
						i_tmp = height - 1
					if (j_tmp < 0): 
						j_tmp = 0
					elif (j_tmp >= width):
						j_tmp = width - 1
					val += kernel[n][m] * int(src[i_tmp][j_tmp])
			if (val >= threshold):
				log[i][j] = 1
			elif (val <= threshold * -1):
				log[i][j] = -1
			else:
				log[i][j] = 0
	return log

def zero_cross(result, height, width, ker_row, ker_col):
	row_size = height + ker_row - 1
	col_size = width + ker_col - 1
	row_frame = ker_row // 2
	col_frame = ker_col // 2
	border_result = np.zeros((row_size, col_size), dtype = 'int8')
	for i in range(row_size):
		for j in range(col_size):
			m = i - row_frame
			n = j - col_frame
			if (m < 0):
				m = 0
			elif (m >= height):
				m = height - 1
			if (n < 0):
				n = 0
			elif (n >= width):
				n = height - 1
			border_result[i, j] = result[m, n]
	edge_img = np.zeros((height, width), dtype = 'uint8')
	for i in range(height):
		for j in range(width):
			if (result[i, j] == 1):
				cross = 0
				for m in range(ker_row):
					for n in range(ker_col):
						if (border_result[i + m, j + n] == -1):
							cross = 1
				if (cross == 1):
					edge_img[i, j] = 255
				else:
					edge_img[i, j] = 0
			else:
				edge_img[i, j] = 0
	return edge_img
'''
log_9_500 = LaplacianofGaussian(img1, LoG_kernel_9, 9, img1_height, img1_width, 500)
img8 = zero_cross(log_9_500, img1_height, img1_width, 3, 3)
cv2.imwrite("result2_9x9_500.jpg", img8)
log_9_1000 = LaplacianofGaussian(img1, LoG_kernel_9, 9, img1_height, img1_width, 1000)
img9 = zero_cross(log_9_1000, img1_height, img1_width, 3, 3)
cv2.imwrite("result2_9x9_1000.jpg", img9)
log_11_3500 = LaplacianofGaussian(img1, LoG_kernel_11, 11, img1_height, img1_width, 3500)
img10 = zero_cross(log_11_3500, img1_height, img1_width, 3, 3)
cv2.imwrite("result2_11x11_3500.jpg", img10)
log_11_4000 = LaplacianofGaussian(img1, LoG_kernel_11, 11, img1_height, img1_width, 4000)
img11 = zero_cross(log_11_4000, img1_height, img1_width, 3, 3)
cv2.imwrite("result2_11x11_4000.jpg", img11)
'''
img11 = zero_cross(log_11_4000, img1_height, img1_width, 3, 3)
cv2.imwrite("result2.jpg", img11)

### Canny
Gaussian_filter = np.array([[2, 4, 5, 4, 2],\
							[4, 9, 12, 9, 4],\
							[5, 12, 15, 12, 5],\
							[4, 9, 12, 9, 4],\
							[2, 4, 5, 4, 2]])
Gaussian_filter = Gaussian_filter / 159
# print(Gaussian_filter)

def noise_reduce(src, filter, filter_size, height, width):
	dst = np.zeros((height, width), dtype = 'uint8')
	border_size = round(filter_size / 2)
	for i in range(height):
		for j in range(width):
			val = 0
			for m in range(filter_size):
				for n in range(filter_size):
					i_tmp = i + m - border_size
					j_tmp = j + n - border_size
					if (i_tmp < 0): 
						i_tmp = 0
					elif (i_tmp >= height):
						i_tmp = height - 1
					if (j_tmp < 0): 
						j_tmp = 0
					elif (j_tmp >= width):
						j_tmp = width - 1
					val += filter[m][n] * int(src[i_tmp][j_tmp])
			dst[i][j] = val
	return dst

def gradient_compute(k, src, height, width):
	row_magnitude = np.zeros((height, width))
	col_magnitude = np.zeros((height, width))
	row_filter = np.array([[-1, 0, 1], [-1 * k, 0, k], [-1, 0, 1]])
	col_filter = np.array([[-1, -1 * k, -1], [0, 0, 0], [1, k, 1]])
	for i in range (height):
		for j in range (width):
			row_val = 0
			col_val = 0
			for m in range(3):
				for n in range(3):
					if (i + m - 1 >= 0 and i + m - 1 < height and j + n - 1 >= 0 and j + n - 1 < width):
						row_val += int(src[i + m - 1][j + n - 1]) * row_filter[m][n]
						col_val += int(src[i + m - 1][j + n - 1]) * col_filter[m][n]
					else:
						row_val += int(src[i][j]) * row_filter[m][n]
						col_val += int(src[i][j]) * col_filter[m][n]
			row_magnitude[i][j] = row_val
			col_magnitude[i][j] = col_val
	dst = np.hypot(row_magnitude, col_magnitude)
	dst = dst / dst.max() * 255
	theta = np.arctan2(col_magnitude, row_magnitude)
	return (dst, theta)

def non_maximal_suppression(grad, theta, height, width):
	dst = np.zeros((height, width), dtype = 'uint8')
	angle = theta * 180.0 / np.pi
	angle[angle < 0] += 180.0
	for i in range(1, height - 1):
		for j in range(1, width - 1):
			l = 0
			p = 0
			#angle 0
			if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
				l = grad[i][j + 1]
				p = grad[i][j - 1]
			#angle 45
			elif (22.5 <= angle[i][j] < 67.5):
				l = grad[i + 1][j - 1]
				p = grad[i - 1][j + 1]
			#angle 90
			elif (67.5 <= angle[i][j] < 112.5):
				l = grad[i + 1][j]
				p = grad[i - 1][j]
			#angle 135
			elif (112.5 <= angle[i][j] < 157.5):
				l = grad[i - 1][j - 1]
				p = grad[i + 1][j + 1]

			if (grad[i][j] >= l) and (grad[i][j] >= p):
				dst[i][j] = grad[i][j]
			else:
				dst[i][j] = 0

	return dst
	
def hysteretic_thresholding(src, height, width, low_thres, high_thres):
	label = np.zeros((height, width), dtype = 'uint8')
	for i in range(height):
		for j in range(width):
			if (src[i][j] >= high_thres):
				label[i][j] = 2
			elif (src[i][j] >= low_thres):
				label[i][j] = 1
			else:
				label[i][j] = 0
	return label

def check_nighbor_label(label, height, width):
	result = np.zeros((height, width), dtype = 'uint8')
	for i in range(height):
		for j in range(width):
			edge_point = 0
			if (label[i][j] == 2):
				edge_point = 255
			elif (label[i][j] == 1):
				for m in range(3):
					if (edge_point == 255):
						break
					for n in range(3):
						if (edge_point == 255):
							break
						if (i + m - 1 >= 0 and i + m - 1 < height and j + n - 1 >= 0 and j + n - 1 < width):
							if (label[i + m - 1][j + n - 1] == 2):
								edge_point = 255
			result[i][j] = edge_point
	return result

def Canny_edge_detect(img, img_height, img_width, low_thres, high_thres):
	img_reduce = noise_reduce(img, Gaussian_filter, 5, img_height, img_width)
	img_grad, img_theta = gradient_compute(2, img_reduce, img_height, img_width)
	img_nms = non_maximal_suppression(img_grad, img_theta, img_height, img_width)
	img_label = hysteretic_thresholding(img_nms, img_height, img_width, low_thres, high_thres)
	img_edgemap = check_nighbor_label(img_label, img_height, img_width)
	return img_edgemap
'''
img12 = Canny_edge_detect(img1, img1_height, img1_width, 20, 40)
cv2.imwrite("result3_20_40.jpg", img12)
img13 = Canny_edge_detect(img1, img1_height, img1_width, 20, 60)
cv2.imwrite("result3_20_60.jpg", img13)
img14 = Canny_edge_detect(img1, img1_height, img1_width, 20, 85)
cv2.imwrite("result3_20_80.jpg", img14)
'''
img12 = Canny_edge_detect(img1, img1_height, img1_width, 10, 30)
cv2.imwrite("result3.jpg", img12)

## Problem 1-b

img16 = cv2.imread('sample2.jpg', cv2.IMREAD_GRAYSCALE)
img16_height = img16.shape[0]
img16_width = img16.shape[1]

filter1 = np.ones((3, 3))
filter1[0][1] = filter1[1][0] = filter1[1][2] = filter1[2][1] = 2
filter1[1][1] = 4
filter1 /= 16

img17 = np.zeros((img16_height, img16_width), dtype = 'uint8')
for i in range (img16_height):
	for j in range (img16_width):
		val = 0
		for m in range(3):
			for n in range(3):
				if (i + m - 1 > 0 and i + m - 1 < img16_height and j + n - 1 > 0 and j + n - 1 < img16_width):
					val += int(img16[i + m - 1][j + n - 1]) * filter1[m][n]
				else:
					val += int(img16[i][j]) * filter1[m][n]
		img17[i][j] = val

c = 0.6
img18 = (c / (2 * c - 1)) * img16 - ((1 - c) / (2 * c - 1)) * img17
cv2.imwrite("result4.jpg", img18)
img19 = first_order_edge(1, img16, img16_height, img16_width, 20)
cv2.imwrite("sample2_edgemap.jpg", img19)
img20 = first_order_edge(1, img18, img16_height, img16_width, 20)
cv2.imwrite("result4_edgemap.jpg", img20)

## Problem 2-a
img21 = cv2.imread('sample3.jpg', cv2.IMREAD_GRAYSCALE)
img21_height = img21.shape[0]
img21_width = img21.shape[1]

img22 = np.zeros((img21_height, img21_width), dtype = 'uint8')
for i in range(img21_height):
	for j in range(img21_width):
		img22[i][j] = img21[round(5 * i / 9)][round(5 * j / 9)]
cv2.imwrite("result5.jpg", img22)

## Problem 2-b
img23 = cv2.imread('sample5.jpg', cv2.IMREAD_GRAYSCALE)
img24 = cv2.imread('sample6.jpg', cv2.IMREAD_GRAYSCALE)

img23_height = img23.shape[0]
img23_width = img23.shape[1]
img25 = np.zeros((img23_height, img23_width), dtype = 'uint8')

for i in range(img23_height):
	for j in range(img23_width):
		try:
			i_tmp = round(i + math.sin((j / 120) * 2 * math.pi) * 40)
			j_tmp = round(j + math.sin((i / 360) * 2 * math.pi) * 60)
			# print(i, i_tmp, j, j_tmp)
			img25[i][j] = img23[i_tmp][j_tmp]
		except IndexError as error:
			pass
cv2.imwrite("result6.jpg", img25)