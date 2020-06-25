import numpy as np
import math
from numba import jit

@jit
def PSNR(noise, original):
	height = original.shape[0]
	width = original.shape[1]
	# print(noise.shape, original.shape)
	MSE = 0
	for i in range(height):
		for j in range(width):
			MSE += math.pow(int(noise[i][j]) - int(original[i][j]), 2)
	MSE /= (height * width)
	return (10 * math.log(255 * 255 / MSE, 10))
