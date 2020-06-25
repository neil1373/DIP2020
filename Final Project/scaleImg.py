import numpy as np
from numba import jit

@jit
def scaleImg(img, p = 3):
	m, n = img.shape
	output = np.zeros((m*3, n*3))

	for j in range(m*3):
		for k in range(n*3):
			output[j][k] = img[j//3][k//3]

	return output
