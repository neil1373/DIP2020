import numpy as np
import cv2
import os
from numba import jit
from error_diffusion import error_diffusion
from scaleImg import scaleImg
from distributionOfSIP import SIPassignment
from combineShares import combineShares
from PSNR import PSNR

@jit
def main():
	img = cv2.imread("sample/lena.png", cv2.IMREAD_GRAYSCALE)
	img_share1 = cv2.imread("sample/baboon.png", cv2.IMREAD_GRAYSCALE)
	img_share2 = cv2.imread("sample/moon.png", cv2.IMREAD_GRAYSCALE)

	# scale original share images to 3x
	img_share1S = scaleImg(img_share1)
	img_share2S = scaleImg(img_share2)
	img_share3S = 255 - img_share2S

	cv2.imwrite("output/baboon_scale.png", img_share1S)
	cv2.imwrite("output/moon_scale.png", img_share2S)
	cv2.imwrite("output/moonContrast_scale.png", img_share3S) 

	k = 3
	n = 3
	q = 9
	r = 4

	img_scale3x = scaleImg(img)
	cv2.imwrite("output/input_grayscale3x.png", img_scale3x)
	img = error_diffusion(img)
	cv2.imwrite("output/input_Halfton.png", img)

	cv_img = np.zeros((img_share1S.shape))
	cv_img += r * 255 / q 
	cv_img = error_diffusion(cv_img, r, True)
	cv2.imwrite("output/cvImg.png", cv_img)

	SIPassignment(img, img_share1S, img_share2S, img_share3S, cv_img)

	cv2.imwrite("output/baboon_SIP.png", img_share1S)
	cv2.imwrite("output/moon_SIP.png", img_share2S)
	cv2.imwrite("output/moonContrast_SIP.png", img_share3S) 

	img_share1H = error_diffusion(img_share1S)
	img_share2H = error_diffusion(img_share2S)
	img_share3H = error_diffusion(img_share3S)

	cv2.imwrite("output/baboon_output.png", img_share1H)
	cv2.imwrite("output/moon_output.png", img_share2H)
	cv2.imwrite("output/moonContrast_output.png", img_share3H) 

	output = combineShares(img_share1S, img_share2S, img_share3S)
	cv2.imwrite("output/secretImg.png", output)

	print("PSNR:", PSNR(output, img_scale3x), "dB")
	print("PSNR:", PSNR(output, img_share1S), "dB")
	print("PSNR:", PSNR(output, img_share2S), "dB")
	print("PSNR:", PSNR(output, img_share3S), "dB")

if __name__ == "__main__":
	main()