import numpy as np
from numba import jit

@jit
def count(img, i, j):
	ret = 0
	for x in range(3):
		for y in range(3):
			if (img[i+x][j+y] == 255):
				ret += 1
	return ret

@jit
def error_diffusion(img, r = 4, ensure = False):
	img_height, img_width = img.shape
	thres = 0.5
	F = img / 255
	total_error = np.zeros((img_height, img_width))
	floyd_mask = np.array([[0, 0, 0], [0, 0, 7], [3, 5, 1]]) / 16
	error = np.zeros((img_height, img_width))

	
	for i in range(img_height):
		for j in range(img_width):
			if (F[i,j] >= 0.5):
				error[i,j] = F[i,j] - 1
			else:
				error[i,j] = F[i,j]
			for m in range(3):
				for n in range(3):
					# try:
					if ((i + m - 1) >= 0 and (i + m - 1) < img_height and (j + n - 1) >= 0 and (j + n - 1) < img_width):
						F[i+m-1, j+n-1] += error[i,j] * floyd_mask[m,n]
					# except:
						pass

	G = np.zeros((img_height, img_width))

	for i in range(img_height):
		for j in range(img_width):
			if (F[i,j] >= 0.5):
				G[i][j] = 255

	if (ensure):
		for i in range(0, img_height, 3):
			for j in range(0, img_width, 3):
				SIP = count(G, i, j)
				flag = False
				if (SIP < 4):
					# constrained to 1s
					for x in range(i+2, i-1, -1):
						if (flag):
							break
						for y in range(j+2, j-1, -1):
							if (G[x][y] == 0):
								G[x][y] = 255
								SIP += 1
								if (SIP == 4):
									flag = True
									break

				elif (SIP > 4):
					# constrained to 0s
					for x in range(i+2, i-1, -1):
						if (flag):
							break
						for y in range(j+2, j-1, -1):
							if (G[x][y] == 255):
								G[x][y] = 0
								SIP -= 1
								if (SIP == 4):
									flag = True
									break	

	return G


