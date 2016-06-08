#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple implementation of the fastest FFT convolution algorithm, as well as
several others.

O(N^2*log(n)) time, where N^2 is the size of the image (we can never get below 
this of course, but we may _approach_ it).
"""

import numpy as np
from numpy.fft import fft2, fftn, ifft2, ifftn
from PIL import Image
from tqdm import tqdm

import kernel

class convolve(object):
    """ contains methods to convolve two images with different convolution 
    algorithms """

    def __init__(self, image_array, kernel):
        self.array = image_array
        self.kernel = kernel

        self.__rangeX_ = self.array.shape[0]
        self.__rangeY_ = self.array.shape[1]

        self.__rangeKX_ = self.kernel.shape[0]
        self.__rangeKY_ = self.kernel.shape[1]

        diffX = self.__rangeX_ % self.__rangeKX_
        diffY = self.__rangeY_ % self.__rangeKY_

    def spaceConv(self):
        """ normal convolution, O(N^2*n^2). This is usually too slow """

        def spot_convolve(subset):
            """ perform a simple 'dot product' between the 2 dimensional image
            subsets. This is the O(n^2) part of the time complexity """
            total = 0.0
            for i in xrange(subset.shape[0]):
                for j in xrange(subset.shape[1]):
                    total += self.kernel[i][j]*subset[i][j]
            return total

        # pad array for convolution
        offsetX = self.__rangeKX_ // 2
        offsetY = self.__rangeKY_ // 2

        self.array = np.pad(self.array,               \
            [(offsetY, offsetY), (offsetX, offsetX)], \
               mode='constant', constant_values=0)

        """ this is the O(N^2) part of this algorithm """
        for i in tqdm(xrange(self.__rangeX_ - 2*offsetX)):
            for j in xrange(self.__rangeY_ - 2*offsetY):
                self.array[i+offsetX][j+offsetY] = \
                  spot_convolve(self.array[i:i+2*offsetX+1,j:j+2*offsetY+1])

        return self.array

    def OaAconvolve(self):
        """ faster convolution algorithm, O(N^2*log(n)). """
        
        def partition():
            """ solve for the divisibility and pad. The biggest problem here
            is ensuring that the partitioning scheme is correct for the overlap
            and add portion of the algorithm. """

            # solve for the total padding along each axis
            diffX = (self.__rangeKX_ - \
                        (self.__rangeX_ - self.__rangeKX_*(\
                        self.__rangeX_ // self.__rangeKX_)))\
                        % self.__rangeKX_
            
            diffY = (self.__rangeKY_ - \
                        (self.__rangeY_ - self.__rangeKY_*(\
                        self.__rangeY_ // self.__rangeKY_)))\
                        % self.__rangeKY_

            # each side, i.e. left, right, top and bottom
            right = diffX // 2
            left = diffX - right
            bottom = diffY // 2
            top = diffY - bottom

            # pad the array now
            self.array = np.pad(self.array, [(left, right), (top, bottom)], \
                mode='constant', constant_values=0)

        def divide_and_transform():
            """ take the padded array and divide it up into kernel-sized
            chunks, pad those chunks and transform both the chunks and the
            kernel with the FFT. """
            self.split = np.split(self.array, self.__rangeX_)

        partition()
        divide_and_transform()

    def OaS_FFT(self):
        """ use the FFT and a partitioning scheme to convolve the image and the
        kernel. In total, time complexity is O(N^2*log(n)) """

        def partitioner():
            """ solve for the best partitions for an image given the image 
            and kernel dimensions """
            blocksizeX = -(-self.rangeX**2 // self.kernel.shape[0]**2)
            blocksizeY = -(-self.rangeYw32**2 // self.kernel.shape[1]**2)


if __name__ == '__main__':
    try:
        import pyplot as plt
    except ImportError:
        import matplotlib.pyplot as plt

    image = np.array(\
        Image.open('/home/brandon/Pictures/lune_2010-09-29_06-09-07-s1.jpg'))

    image = image.T[0]

    kern = kernel.Kernel()
    kern = kern.Kg2(15, 15, sigma=1.5, muX=0.0, muY=0.0)
    kern /= np.sum(kern)        # normalize volume
    plt.imshow(kern, interpolation='none', cmap='gist_heat')
    plt.colorbar()
    plt.show()

    conv = convolve(image, kern)

    plt.imshow(conv.OaAconvolve(), interpolation='none', cmap='gray')
    plt.show()