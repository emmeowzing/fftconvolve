#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A quick example :P """

from fftconv import *
from scipy.misc import imsave, imread
import _kernel
import numpy as np
import matplotlib.pyplot as plt


def main():
	## Initialize a kernel
	kern = _kernel.Kernel()
	kern = kern.Kg2(13, 13, sigma=3.5, muX=0.0, muY=0.0)
	kern /= np.sum(kern)	# normalize volume

	## open an image
	image = imread('spider.jpg').T[0]

	## Convolve using some method
	conv = convolve(image, kern)
	plt.imshow(conv.OAconv2(), interpolation='none', cmap='gray')
	plt.show()


if __name__ == '__main__':
	main()