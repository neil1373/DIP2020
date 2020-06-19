import numpy as np
np.set_printoptions(suppress=True)
import cv2
import matplotlib.pyplot as plt
import math
import random
import os
#from numba import jit

#@jit
def border_extend(img, border_size):
	height = img.shape[0]
	width = img.shape[1]
	border_img_height = height + 2 * border_size
	border_img_width = width + 2 * border_size
	border_img = np.zeros((border_img_height, border_img_width))
	for i in range(border_img_height):
		for j in range(border_img_width):
			idx_i = i - border_size
			if (idx_i < 0):
				idx_i = -1 * idx_i
			elif (idx_i >= height):
				idx_i = 2 * height - 2 - idx_i
			idx_j = j - border_size
			if (idx_j < 0):
				idx_j = -1 * idx_j
			elif (idx_j >= width):
				idx_j = 2 * width - 2 - idx_j
			border_img[i][j] = img[idx_i][idx_j]
	return border_img

#@jit
def K_means_cluster(texture_data, k):
	height = texture_data.shape[1]
	width = texture_data.shape[2]
	label = np.zeros((height, width), dtype = 'int8')
	prev_label = np.ones((height, width), dtype = 'int8')
	center = np.zeros((k, 2), dtype = 'int32')
	center_texture_vector = np.zeros((k, 9), dtype = 'float64')
	'''
	center[0] = np.array([random.randint(240, 380), random.randint(200, 650)])
	center[1] = np.array([random.randint(550, height - 1), random.randint(0, width - 1)])
	center[2] = np.array([random.randint(0, 140), random.randint(0, width - 1)])
	'''
	center[0] = np.array([352, 500])
	center[1] = np.array([653, 500])
	center[2] = np.array([51, 500])
	'''
	for n in range(k):
		center[n] = np.array([random.randint(0, height - 1), random.randint(0, width - 1)])
	'''
	diff = np.sum(np.abs(label - prev_label))
	# print("diff =", diff)
	weight = np.ones((9), dtype = 'int32')
	# weight[0] = weight[3] = weight[6] = 4
	iter = 0
	if os.path.exists("iteration") == False:
		os.mkdir("iteration")
	while (diff != 0 or iter < 10):
		iter += 1
		print("\t\t Iteration" , iter)
		for n in range(k):
			center_texture_vector[n] = texture_data[:, np.int(center[n][0]), np.int(center[n][1])]
		# print(center)
		# print(center_texture_vector)
		
		points_sum = np.zeros((k, 2), dtype = 'int32')
		points_num = np.zeros((k), dtype = 'int32')
		for i in range(height):
			for j in range(width):
				target_texture_vector = texture_data[:, i, j]
				target_dist = np.zeros((k))
				for n in range(k):
					target_dist[n] = np.sum(np.dot(np.power((target_texture_vector - center_texture_vector[n]), 2), weight))
				target_label = np.argmin(target_dist)
				label[i][j] = target_label
				points_sum[target_label] += [i, j]
				points_num[target_label] += 1
		filename = "./iteration/sample1_label_{count}.jpg".format(count = iter)
		cv2.imwrite(filename, label.astype('uint8') * 127)
		# print(points_num)
		center = (points_sum.T / points_num.T).T
		diff = np.sum(np.abs(np.subtract(label, prev_label)))
		# print("diff =", diff)
		np.copyto(prev_label, label)
	return label.astype('uint8')

# Connected components Labeling
#@jit
def object_segment(train_set, H, W):
	labels = np.zeros((H, W), dtype = 'int32') # image labels
	count = 0
	# segments = np.ones((36, 75, 90), dtype = 'uint8') * 255
	box_edge = np.zeros((40, 4), dtype = 'int32')
	# binary image and image label initialization of training set
	for i in range(H):
		for j in range(W):
			val = train_set[i, j]
			if train_set[i, j] <= 127:
				count += 1
				labels[i, j] = count
				if train_set[i, j] <= 127:
					train_set[i, j] = 0
				else:
					train_set[i, j] = 255
	parent = np.arange(count + 100)
	change = 1
	while (change):
		change = 0
		# Top-down scaning
		for i in range(H):
			for j in range(W):
				if labels[i, j] != 0:
					minlabel = labels[i, j]
					check = 0
					# check pixel upside
					if i != 0:
						if (0 < labels[(i - 1), j] < minlabel) and check != 1:
							minlabel = labels[(i - 1), j]
							check = 1
					# check pixel leftside
					if j != 0:
						if (0 < labels[i, (j - 1)] < minlabel) and check != 1:
							minlabel = labels[i ,(j - 1)]
							check = 1
					# adjust label of current pixel
					if minlabel != labels[i, j]:
						change = 1
						tag = labels[i, j]
						parent[tag] = minlabel
						labels[i, j] = minlabel
		# Bottom-up scaning
		for i in range(H - 1, -1, -1):
			for j in range(W - 1, -1, -1):
				if labels[i, j] != 0:
					minlabel = labels[i, j]
					check = 0
					# check pixel downside
					if i != H - 1:
						if (0 < labels[(i + 1), j] < minlabel) and check != 1:
							minlabel = labels[(i + 1), j]
							check = 1
					# check pixel rightside
					if j != W - 1:
						if (0 < labels[i, (j + 1)] < minlabel) and check != 1:
							minlabel = labels[i ,(j + 1)]
							check = 1
					# adjust label of current pixel
					if minlabel != labels[i, j]:
						change = 1
						tag = labels[i, j]
						parent[tag] = minlabel
						labels[i, j] = minlabel

	# count label frequency
	lab_fq = np.zeros((count), dtype = np.int32)
	for i in range(H):
		for j in range(W):
			labels[i, j] = parent[labels[i, j]]
			lab_fq[labels[i, j]] += 1
	segment_count = 0
	# print(count)
	for k in range(count):
		if k != 0:
			if lab_fq[k] >= 300 and lab_fq[k] < 5000:
				top = H
				bottom = -1
				left = W
				right = -1
				for i in range(H):
					for j in range(W):
						if (labels[i, j] == k):
							if i < top:
								top = i
							if i > bottom:
								bottom = i
							if j < left:
								left = j
							if j > right:
								right = j
				# print(top, bottom, left, right, bottom - top, right - left)
				if ((bottom - top) * (right - left) <= H * W * 0.25):
					box_edge[segment_count] = np.array([top, bottom, left, right])
					# segment = np.array(train_set[top:bottom, left:right])
					# segments[char_count, 0:(bottom - top), 0:(right - left)] = np.copy(segment)
					segment_count += 1

					# filename = "./train_segment/char_{count}.jpg".format(count = segment_count)
					# cv2.imwrite(filename, segments[segment_count - 1])
					# cv2.rectangle(train_set, (left, top), (right, bottom), (127, 0, 0), 3)
					count -= lab_fq[k]
	# cv2.imwrite('TrainingSet_boxes.jpg', train_set) # Draw bounding box
	return box_edge

#@jit
def plate_preprocess(plate):
	if (np.sum(plate) / 255 < plate.shape[0] * plate.shape[1] * 0.5):
		plate = 255 - plate
	plate_boxes = object_segment(plate, plate.shape[0], plate.shape[1])
	amount = 0
	for n in range(40):
		if np.sum(plate_boxes[n]) == 0:
			amount = n
			break
	## sort bounding boxes
	for i in range(amount):
		for j in range(i, amount):
			if (plate_boxes[i, 2] > (plate_boxes[j, 2] + 10) and plate_boxes[i, 3] > (plate_boxes[j, 3] + 10)):
				if (plate_boxes[i, 0] < (plate_boxes[j, 0] + 10) and plate_boxes[i, 1] < (plate_boxes[j, 1] - 10)):
					pass
				else:
					tmp = np.copy(plate_boxes[i])
					plate_boxes[i] = np.copy(plate_boxes[j])
					plate_boxes[j] = np.copy(tmp)
	for i in range(amount - 1):
		if (plate_boxes[i, 3] >= plate_boxes[i + 1, 2]):
			if (plate_boxes[i, 0] < (plate_boxes[j, 0] + 10) and plate_boxes[i, 1] < (plate_boxes[j, 1] - 10)):
				pass
			else:
				overlap = plate_boxes[i, 3] - plate_boxes[i + 1, 2] + 1
				plate_boxes[i, 3] -= overlap
				plate_boxes[i + 1, 2] += overlap
	plate_sorted_boxes = np.copy(plate_boxes[0:amount])
	return plate, plate_sorted_boxes

#@jit
def segment_attributes(img, img_boxes):
	attributes = np.zeros((img_boxes.shape[0], 10), dtype = 'int32')
	objects = img_boxes.shape[0]
	for n in range(objects):
		# print(train_set_edges[n])
		[top, bottom, left, right] = img_boxes[n]
		segment = np.copy(img[top:bottom+1, left:right+1])
		floor = 0
		seg_h, seg_w = (bottom - top), (right - left)
		mid_hor = seg_h // 2
		quad_hor = 2 * seg_h // 5
		quad3_hor = 3 * seg_h // 5
		mid_ver = seg_w // 2
		quad_ver = 2 * seg_w // 5
		quad3_ver = 3 * seg_w // 5
		if (objects == 7):	# Plate Case
			floor += 3
			seg_h -= 3
			seg_w -= 3
		for i in range(seg_h - 1):
			if (segment[i][floor] != segment[i + 1][floor]):
				attributes[n][0] += 1
			if (segment[i][quad_ver] != segment[i + 1][quad_ver]):
				attributes[n][1] += 1
			if (segment[i][mid_ver] != segment[i + 1][mid_ver]):
				attributes[n][2] += 1
			if (segment[i][quad3_ver] != segment[i + 1][quad3_ver]):
				attributes[n][3] += 1
			if (segment[i][seg_w] != segment[i + 1][seg_w]):
				attributes[n][4] += 1
		for j in range(seg_w - 1):
			if (segment[floor][j] != segment[floor][j + 1]):
				attributes[n][5] += 1
			if (segment[quad_hor][j] != segment[quad_hor][j + 1]):
				attributes[n][6] += 1
			if (segment[mid_hor][j] != segment[mid_hor][j + 1]):
				attributes[n][7] += 1
			if (segment[quad3_hor][j] != segment[quad3_hor][j + 1]):
				attributes[n][8] += 1
			if (segment[seg_h][j] != segment[seg_h][j + 1]):
				attributes[n][9] += 1
	return attributes

def plate_recognition(train_attributes, plate_attributes):
	char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',\
				'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',\
				'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	plate_list = []
	for n in range(7):
		if (n < 3):		#alphabet
			dist = np.abs(train_attributes[0:26] - plate_attributes[n])
			mag_dist = np.sum(dist, axis = 1)
			euc_dist = np.sum(np.power(dist, 2), axis = 1)
			# print(mag_dist)
			# print(euc_dist)
			# print(char_list[np.argmin(mag_dist)], char_list[np.argmin(euc_dist)])
			plate_list.append(char_list[25 - np.argmin(np.flip(euc_dist))])
		else:
			dist = np.abs(train_attributes[26:36] - plate_attributes[n])
			mag_dist = np.sum(dist, axis = 1)
			euc_dist = np.sum(np.power(dist, 2), axis = 1)
			# print(mag_dist)
			# print(euc_dist)
			# print(char_list[np.argmin(mag_dist) + 26], char_list[np.argmin(euc_dist) + 26])
			plate_list.append(char_list[35 - np.argmin(np.flip(euc_dist))])
	return plate_list
#@jit
def main():
	## Problem 1
	print("[Problem 1]")
	# string = "sample1_label_{count}.jpg".format(count = count)
	src_img = cv2.imread('sample1.jpg', cv2.IMREAD_GRAYSCALE)
	height = src_img.shape[0]
	width = src_img.shape[1]

	base_vecs = np.zeros((3, 3))
	base_vecs[0] = np.array([1, 2, 1]) / 6		# Local averaging
	base_vecs[1] = np.array([-1, 0, 1]) / 2		# Edge detector
	base_vecs[2] = np.array([1, -2, 1]) / 2		# Spot detector
	mask = np.zeros((9, 3, 3))
	for i in range(3):
		for j in range(3):
			mask[3 * i + j] = np.outer(base_vecs[i], base_vecs[j])
	src_img_extend = border_extend(src_img, 1)
	# tmp = src_img_extend[0:3, 0:3]
	print("\t Building microstructure...")
	microstructure = np.zeros((9, height, width), dtype = 'float64')
	for i in range(height):
		for j in range(width):
			for m in range(9):
				tmp = src_img_extend[i:i+3, j:j+3]
				microstructure[m][i][j] = np.sum(np.multiply(tmp, mask[m]))

	window_size = 19
	print("\t Extending microstructure...")
	microstructure_extend = np.zeros((9, height + window_size - 1, width + window_size - 1), dtype = 'float64')
	for m in range(9):
		microstructure_extend[m] = border_extend(microstructure[m], window_size // 2)

	print("\t Getting texture data...")
	texture_data = np.zeros((9, height, width), dtype = 'float64')
	for i in range(height):
		for j in range(width):
			for m in range(9):
				window = microstructure_extend[m][i:i+window_size, j:j+window_size]
				texture_data[m][i][j] = np.sum(np.power(window, 2))

	print("\t k-means clustering...")
	texture_label = K_means_cluster(texture_data, 3)
	cv2.imwrite("sample1_label_final.jpg", texture_label * 127)
	
	## Problem 2
	print("[Problem 2]")
	print("\t Processing TrainingSet.jpg\n")
	train_set = cv2.imread('TrainingSet.jpg', cv2.IMREAD_GRAYSCALE)
	train_set = cv2.threshold(train_set, 128, 255, cv2.THRESH_BINARY)[1] # Binarize
	train_set, train_set_edges = plate_preprocess(train_set)
	train_attributes = segment_attributes(train_set, train_set_edges)
	if os.path.exists("train_segments") == False:
		os.mkdir("train_segments")
	for n in range(train_set_edges.shape[0]):
		[top, bottom, left, right] = train_set_edges[n]
		filename = "./train_segments/char_{count}.jpg".format(count = n)
		cv2.imwrite(filename, train_set[top:bottom+1, left:right+1])
		cv2.rectangle(train_set, (left, top), (right, bottom), (127, 0, 0), 2)
	cv2.imwrite("TrainingSet_boxes.jpg", train_set)

	# print(train_attributes)
	file_idx = 2
	input_file = "sample{idx}.jpg".format(idx = file_idx)
	print("\t Processing " + input_file)
	plate_1 = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
	plate_1 = cv2.threshold(plate_1, 128, 255, cv2.THRESH_BINARY)[1]	#Binarize
	plate_1, plate_1_boxes = plate_preprocess(plate_1)
	plate_1_attributes = segment_attributes(plate_1, plate_1_boxes)
	plate_1_list = plate_recognition(train_attributes, plate_1_attributes)
	plate_1_str = ""
	plate_1_str = plate_1_str.join(plate_1_list)
	# print(plate_1_attributes)
	print("\t\t\t" + plate_1_str)
	if os.path.exists("plate1_segments") == False:
		os.mkdir("plate1_segments")
	for n in range(7):
		# print(plate_1_boxes[n])
		[top, bottom, left, right] = plate_1_boxes[n]
		filename = "./plate1_segments/char_{count}.jpg".format(count = n)
		cv2.imwrite(filename, plate_1[top:bottom, left:right])
		cv2.rectangle(plate_1, (left, top), (right, bottom), (127, 0, 0), 2)
	cv2.imwrite("sample2_binary.jpg", plate_1)	

	file_idx += 1
	input_file = "sample{idx}.jpg".format(idx = file_idx)
	print("\t Processing " + input_file)
	plate_2 = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
	plate_2 = cv2.threshold(plate_2, 128, 255, cv2.THRESH_BINARY)[1]	#Binarize
	plate_2, plate_2_boxes = plate_preprocess(plate_2)
	plate_2_attributes = segment_attributes(plate_2, plate_2_boxes)
	plate_2_list = plate_recognition(train_attributes, plate_2_attributes)
	plate_2_str = ""
	plate_2_str = plate_2_str.join(plate_2_list)
	# print(plate_2_attributes)
	print("\t\t\t" + plate_2_str)
	if os.path.exists("plate2_segments") == False:
		os.mkdir("plate2_segments")
	for n in range(7):
		# print(plate_2_boxes[n])
		[top, bottom, left, right] = plate_2_boxes[n]
		filename = "./plate2_segments/char_{count}.jpg".format(count = n)
		cv2.imwrite(filename, plate_2[top:bottom, left:right])
		cv2.rectangle(plate_2, (left, top), (right, bottom), (127, 0, 0), 2)
	cv2.imwrite("sample3_binary.jpg", plate_2)

	file_idx += 1
	input_file = "sample{idx}.jpg".format(idx = file_idx)
	print("\t Processing " + input_file)
	plate_3 = cv2.imread('sample4.jpg', cv2.IMREAD_GRAYSCALE)
	plate_3 = cv2.threshold(plate_3, 128, 255, cv2.THRESH_BINARY)[1]	#Binarize
	plate_3, plate_3_boxes = plate_preprocess(plate_3)
	plate_3_attributes = segment_attributes(plate_3, plate_3_boxes)
	plate_3_list = plate_recognition(train_attributes, plate_3_attributes)
	plate_3_str = ""
	plate_3_str = plate_3_str.join(plate_3_list)
	# print(plate_3_attributes)
	print("\t\t\t" + plate_3_str)
	if os.path.exists("plate3_segments") == False:
		os.mkdir("plate3_segments")
	for n in range(7):
		# print(plate_3_boxes[n])
		[top, bottom, left, right] = plate_3_boxes[n]
		filename = "./plate3_segments/char_{count}.jpg".format(count = n)
		cv2.imwrite(filename, plate_3[top:bottom, left:right])
		cv2.rectangle(plate_3, (left, top), (right, bottom), (127, 0, 0), 2)
	cv2.imwrite("sample4_binary.jpg", plate_3)

if __name__ == '__main__':
	main()