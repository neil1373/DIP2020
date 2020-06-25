import numpy as np
from numba import jit

@jit
def combineShares(share1, share2, share3):
	m, n = share1.shape

	img = np.zeros((m, n))
	output = np.zeros((m, n))
	thres = np.zeros((m, n))
	
	for i in range(m):
		for j in range(n):
			img[i][j] = share1[i][j] + share2[i][j] + share3[i][j]
			img[i][j] /= 255

			# TODO: fix this!
			thres[i][j] = 0.25 + 0.33*0.25*(img[i][j-1]+img[i][j-2]+img[i][j-3])
			if img[i][j] > thres[i][j]:
				output[i][j] = 0
			else:
				output[i][j] = 255





	return output