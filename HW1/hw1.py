import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

# Problem 0-a
img1_tmp = np.fromfile('sample1.raw', dtype = 'uint8')
img1 = img1_tmp.reshape((400, 600))
cv2.imwrite("result1.jpg", img1)
print("[Generate] result1.jpg")
print("[Complete] 0(a)")

# Problem 0-b
img2 = cv2.imread('sample2.jpg')
img2_height = img2.shape[0]
img2_width = img2.shape[1]
img2_gray = np.zeros((img2_height, img2_width), dtype = 'uint8')
for i in range(img2_height):
	for j in range(img2_width):
		img2_gray[i][j] = np.round(img2[i][j][0] * 0.2126 + img2[i][j][1] * 0.7152 + img2[i][j][2] * 0.0722)
cv2.imwrite("result2.jpg", img2_gray)
print("[Generate] result2.jpg")
print("[Complete] 0(b)")

# Problem 0-c
img3 = np.zeros((img2_height, img2_width), dtype = 'uint8')
img4 = np.zeros((img2_height, img2_width), dtype = 'uint8')
for i in range (img2_width):
	for j in range (img2_height):
		img3[i][j] = img2_gray[j][img2_width - 1 - i]
for i in range (img2_height):
	for j in range (img2_width):
		img4[i][j] = img2_gray[j][i]
cv2.imwrite("result3.jpg", img3)
print("[Generate] result3.jpg")
cv2.imwrite("result4.jpg", img4)
print("[Generate] result4.jpg")
print("[Complete] 0(c)")

# Problem 1-a
img5 = cv2.imread('sample3.jpg', cv2.IMREAD_GRAYSCALE)
img1_histo = cv2.calcHist([img1], [0], None, [256], [0, 256])
plt.bar(range(1,257), img1_histo, color = ['black'])
plt.savefig("result1_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] result1_histo.jpg")
img5_histo = cv2.calcHist([img5], [0], None, [256], [0, 256])
plt.bar(range(1,257), img5_histo, color = ['black'])
plt.savefig("sample3_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] sample3_histo.jpg")
print("[Complete] 1(a)")

# Problem 1-b
img5_height = img5.shape[0]
img5_width = img5.shape[1]
img5_size = img5_height * img5_width
img5_cdf = np.zeros((256))
sum = 0
for k in range(256):
	tmp = img5_histo[k] / img5_size
	tmp *= 255
	sum += tmp
	img5_cdf[k] = np.round(sum)

img6 = np.zeros((img5_height, img5_width), dtype = 'uint8')
for i in range(img5_height):
    for j in range(img5_width):
        img6[i][j] = img5_cdf[img5[i][j]]

cv2.imwrite("result5.jpg", img6)
print("[Generate] result5.jpg")
img6_histo = cv2.calcHist([img6], [0], None, [256], [0, 256])
sum = 0
for k in range(256):
	sum += img6_histo[k]

plt.bar(range(1,257), img6_histo, color = ['black'])
plt.savefig("result5_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] result5_histo.jpg")
print("[Complete] 1(b)")

# Problem 1-c
img7_13 = np.zeros((img5_height, img5_width), dtype = 'uint8')
img7_15 = np.zeros((img5_height, img5_width), dtype = 'uint8')

for i in range(img5_height):
	for j in range(img5_width):
		'''
		# window size = 13x13
		count_13 = 169
		for m in range(13):
			for n in range(13):
				if ((i + m - 6) >= 0 and (i + m - 6) < img5_height and (j + n - 6) >= 0 and (j + n - 6) < img5_width):
					if (img5[i + m - 6][j + n - 6] > img5[i][j]):
						count_13 -= 1
		img7_13[i][j] = np.round(count_13 / 169 * 255)
		'''
		# window size = 15x15
		count_15 = 225
		for m in range(15):
			for n in range(15):
				if ((i + m - 7) >= 0 and (i + m - 7) < img5_height and (j + n - 7) >= 0 and (j + n - 7) < img5_width):
					if (img5[i + m - 7][j + n - 7] > img5[i][j]):
						count_15 -= 1
		img7_15[i][j] = np.round(count_15 / 225 * 255)
'''
cv2.imwrite("result6_13.jpg", img7_13)
img7_13_histo = cv2.calcHist([img7_13], [0], None, [256], [0, 256])
plt.bar(range(1,257), img7_13_histo, color = ['black'])
plt.savefig("result6_13_histo.jpg", bbox_inches='tight')
plt.close()
'''
cv2.imwrite("result6.jpg", img7_15)
print("[Generate] result6.jpg")
img7_15_histo = cv2.calcHist([img7_15], [0], None, [256], [0, 256])
plt.bar(range(1,257), img7_15_histo, color = ['black'])
plt.savefig("result6_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] result6_histo.jpg")
print("[Complete] 1(c)")

# Problem 1-e (Log Transform)
img8 = np.zeros((img5_height, img5_width), dtype = 'uint8')
for i in range(img5_height):
	for j in range(img5_width):
		img8[i][j] = math.log(img5[i][j] / 255 * 3 + 1) / math.log(2) * 255	# Apply Log Transform
cv2.imwrite("result7.jpg", img8)
print("[Generate] result7.jpg")

img8_histo = cv2.calcHist([img8], [0], None, [256], [0, 256])
plt.bar(range(1,257), img8_histo, color = ['black'])
plt.savefig("result7_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] result7_histo.jpg")

# Problem 1-e (Inverse Log Transform)
img9 = np.zeros((img5_height, img5_width), dtype = 'uint8')
for i in range(img5_height):
	for j in range(img5_width):
		tmp = math.log(img5[i][j] / 255 * 3 + 1) / math.log(2)
		if (tmp < 0.1):
			img9[i][j] = 255
		else:
			img9[i][j] = 0.1 / tmp * 255
cv2.imwrite("result8.jpg", img9)
print("[Generate] result8.jpg")

img9_histo = cv2.calcHist([img9], [0], None, [256], [0, 256])
plt.bar(range(1,257), img9_histo, color = ['black'])
plt.savefig("result8_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] result8_histo.jpg")

# Problem 1-e (Power-law Transform)
img10 = np.zeros((img5_height, img5_width), dtype = 'uint8')
for i in range(img5_height):
	for j in range(img5_width):
		img10[i][j] = math.pow((img5[i][j] / 255), 1/3) * 255	# Apply power law with p = 1/2
cv2.imwrite("result9.jpg", img10)
print("[Generate] result9.jpg")

img10_histo = cv2.calcHist([img10], [0], None, [256], [0, 256])
plt.bar(range(1,257), img10_histo, color = ['black'])
plt.savefig("result9_histo.jpg", bbox_inches='tight')
plt.close()
print("[Generate] result9_histo.jpg")

print("[Complete] 1(e)")
# Problem 2-a
img11 = cv2.imread('sample4.jpg', cv2.IMREAD_GRAYSCALE)
img11_height = img11.shape[0]
img11_width = img11.shape[1]

img12 = np.zeros((img11_height, img11_width), dtype = 'uint8')
img13 = np.zeros((img11_height, img11_width), dtype = 'uint8')
for i in range(img11_height):
	for j in range(img11_width):
		img12[i][j] = img11[i][j] + random.gauss(0, 1) * 10
		img13[i][j] = img11[i][j] + random.gauss(0, 1) * 20

cv2.imwrite("resultG1.jpg", img12)
print("[Generate] resultG1.jpg")
cv2.imwrite("resultG2.jpg", img13)
print("[Generate] resultG2.jpg")
print("[Complete] 2(a)")

# Problem 2-b
img14 = np.zeros((img11_height, img11_width), dtype = 'uint8')
img15 = np.zeros((img11_height, img11_width), dtype = 'uint8')
thres1 = 0.005
thres2 = 0.01

for i in range(img11_height):
	for j in range(img11_width):
		tmp_prob = random.uniform(0, 1)
		if (tmp_prob < thres1):
			img14[i][j] = 0
		elif (tmp_prob > 1 - thres1):
			img14[i][j] = 255
		else:
			img14[i][j] = img11[i][j]
		if (tmp_prob < thres2):
			img15[i][j] = 0
		elif (tmp_prob > 1 - thres2):
			img15[i][j] = 255
		else:
			img15[i][j] = img11[i][j]

cv2.imwrite("resultS1.jpg", img14)
print("[Generate] resultS1.jpg")
cv2.imwrite("resultS2.jpg", img15)
print("[Generate] resultS2.jpg")
print("[Complete] 2(b)")

# Problem 2-c
mask = np.array([[.0625, .125, .0625],[.125, .25, .125],[.0625, .125, .0625]])
print("Mask:")
print(mask)
img16 = np.zeros((img11_height, img11_width), dtype = 'uint8')
img17 = np.zeros((img11_height, img11_width), dtype = 'uint8')
for i in range(img11_height):
	for j in range(img11_width):
		sum1 = 0
		sum2 = 0
		weight = 0
		for m in range(3):
			for n in range(3):
				if ((i + m - 1) >= 0 and (i + m - 1) < img11_height and (j + n - 1) >= 0 and (j + n - 1) < img11_width):
					sum1 += img12[i + m - 1][j + n - 1] * mask[m][n]
					sum2 += img13[i + m - 1][j + n - 1] * mask[m][n]
					weight += mask[m][n]
		img16[i][j] = np.round(sum1 / weight)
		img17[i][j] = np.round(sum2 / weight)
cv2.imwrite("resultR1.jpg", img16)
print("[Generate] resultR1.jpg")
cv2.imwrite("resultR2.jpg", img17)
print("[Generate] resultR2.jpg")
print("[Complete] 2(c)")

# Problem 2-d
img18 = np.zeros((img11_height, img11_width), dtype = 'uint8')
img19 = np.zeros((img11_height, img11_width), dtype = 'uint8')
epsilon = 65
print("epsilon =", epsilon)
for i in range(img11_height):
	for j in range(img11_width):
		outer_sum_1 = 0
		outer_sum_2 = 0
		outer_amount = 0
		for m in range(3):
			for n in range(3):
				if ((i + m - 1) >= 0 and (i + m - 1) < img11_height and (j + n - 1) >= 0 and (j + n - 1) < img11_width):
					if (m != 1 or n != 1):
						outer_sum_1 += img14[i + m - 1][j + n - 1]
						outer_sum_2 += img15[i + m - 1][j + n - 1]
						outer_amount += 1
		outer_avg_1 = np.round(outer_sum_1 / outer_amount)
		outer_avg_2 = np.round(outer_sum_2 / outer_amount)
		gap_1 = int(img14[i][j]) - int(outer_avg_1)
		gap_2 = int(img15[i][j]) - int(outer_avg_2)
		if gap_1 > epsilon or gap_1 < -1 * epsilon:
			img18[i][j] = outer_avg_1
		else:
			img18[i][j] = img14[i][j]
		if gap_2 > epsilon or gap_2 < -1 * epsilon:
			img19[i][j] = outer_avg_2
		else:
			img19[i][j] = img15[i][j]
cv2.imwrite("resultR3.jpg", img18)
print("[Generate] resultR3.jpg")
cv2.imwrite("resultR4.jpg", img19)
print("[Generate] resultR4.jpg")
print("[Complete] 2(d)")

# Problem 2-e
def PSNR(noise, original):
	height = original.shape[0]
	width = original.shape[1]
	MSE = 0
	for i in range(height):
		for j in range(width):
			MSE += math.pow(int(noise[i][j]) - int(original[i][j]), 2)
	MSE /= (height * width)
	return (10 * math.log(255 * 255 / MSE, 10))

R1_PSNR = PSNR(img16, img11)
R2_PSNR = PSNR(img17, img11)
R3_PSNR = PSNR(img18, img11)
R4_PSNR = PSNR(img19, img11)
print("[Output\t ] R1_PSNR =", R1_PSNR)
print("[Output\t ] R2_PSNR =", R2_PSNR)
print("[Output\t ] R3_PSNR =", R3_PSNR)
print("[Output\t ] R4_PSNR =", R4_PSNR)

# Bonus
img20 = cv2.imread('sample5.jpg', cv2.IMREAD_GRAYSCALE)
img20_height = img20.shape[0]
img20_width = img20.shape[1]
img21 = np.zeros((img20_height, img20_width), dtype = 'uint8')
img22 = np.zeros((img20_height, img20_width), dtype = 'uint8')

for i in range(img20_height):
	for j in range(img20_width):
		tmp_list = []
		for m in range(7):
			for n in range(5):
				if ((i + m - 3) > 0 and (i + m - 3) < img20_height and (j + n - 2) > 0 and (j + n - 2) < img20_width):
					tmp_list.append(img20[i + m - 3][j + n - 2])
		tmp_list.sort()
		count = len(tmp_list)
		if (count % 2 == 0):
			img21[i][j] = round(tmp_list[int(count / 2) - 1] / 2 + tmp_list[int(count / 2)] / 2)
		else:
			img21[i][j] = tmp_list[math.floor(count / 2)]

for i in range(img11_height):
	for j in range(img11_width):
		sum3 = 0
		weight = 0
		for m in range(3):
			for n in range(3):
				if ((i + m - 1) >= 0 and (i + m - 1) < img11_height and (j + n - 1) >= 0 and (j + n - 1) < img11_width):
					sum3 += img21[i + m - 1][j + n - 1] * mask[m][n]
					weight += mask[m][n]
		img22[i][j] = np.round(sum3 / weight)
cv2.imwrite("result_bonus.jpg", img21)
print("[Generate] result_bonus.jpg")
print("[Complete] Bonus")