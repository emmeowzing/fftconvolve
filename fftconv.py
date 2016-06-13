#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple implementation of the fastest FFT convolution algorithm, as well as
several others.

O(N^2*log(n)) time, where N^2 is the size of the image (we can never get below 
this of course, but we may _approach_ it).
"""

import numpy as np
from numpy.fft import fft2 as FFT, ifft2 as IFFT
from PIL import Image
from tqdm import tqdm

from time import sleep

import kernel

class convolve(object):

    """ contains methods to convolve two images """

    def __init__(self, image_array, kernel):
        self.array = image_array
        self.kernel = kernel

        self.__rangeX_ = self.array.shape[0]
        self.__rangeY_ = self.array.shape[1]

        self.__rangeKX_ = self.kernel.shape[0]
        self.__rangeKY_ = self.kernel.shape[1]
    
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
        
        # solve for the total padding along each axis
        diffX = (self.__rangeKX_ - self.__rangeX_ + self.__rangeKX_*(   \
                    self.__rangeX_ // self.__rangeKX_)) % self.__rangeKX_
        
        diffY = (self.__rangeKY_ - self.__rangeY_ + self.__rangeKY_*(   \
                    self.__rangeY_ // self.__rangeKY_)) % self.__rangeKY_

        # padding on each side, i.e. left, right, top and bottom, centered
        # as well as possible
        right = diffX // 2
        left = diffX - right
        bottom = diffY // 2
        top = diffY - bottom

        # pad the array [(top, bottom), (left, right)]
        self.array = np.pad(self.array, [(top, bottom), (left, right)], \
                mode='constant', constant_values=0)

        # a list of tuples to partition the array
        subsets = [(i*self.__rangeKX_, (i + 1)*self.__rangeKX_,         \
                 j*self.__rangeKY_, (j + 1)*self.__rangeKY_)            \
                 for i in xrange(self.array.shape[0] // self.__rangeKX_)\
                 for j in xrange(self.array.shape[1] // self.__rangeKY_)]

        # padding for individual blocks in the subsets list
        padX = self.__rangeKX_ // 2
        padY = self.__rangeKY_ // 2

        self.kernel = np.pad(self.kernel, [(padY, padY), (padX, padX)], \
                    mode='constant', constant_values=0)

        transformed_kernel = FFT(self.kernel)
        
        # create a blank array of the same size to lay them down
        convolved_image = np.zeros([self.array.shape[0] + 2*padX, \
                                    self.array.shape[1] + 2*padY])

        # transform each partition and OaA on the convolved_image
        for tup in tqdm(subsets):
            # slice and pad the array subset
            subset = np.pad(self.array[tup[0]:tup[1], tup[2]:tup[3]], \
                            [(padY, padY), (padX, padX)], \
                            mode='constant', constant_values=0)

            transformed_subset = FFT(subset)

            # multiply the two arrays entrywise and take the IFFT. np.real()
            # is used because some residual/negligible imaginary terms are 
            # left over after the IFFT.
            space = np.real(IFFT(transformed_kernel*transformed_subset))

            # overlap with indices and add them together to build the image
            convolved_image[tup[0]:tup[1] + 2*padX,tup[2]:tup[3] + 2*padY] += space

        # crop image and get it back, convolved
        return convolved_image[padX + left:padX + left + self.__rangeX_,\
                               padY + bottom:padY + bottom + self.__rangeY_]


if __name__ == '__main__':
    try:
        import pyplot as plt
    except ImportError:
        import matplotlib.pyplot as plt

    image = np.array(Image.open('/home/brandon/Pictures/Portal_Companion_Cube.jpg'))

    image = np.rot90(np.rot90(np.rot90(image.T[0])))

    """
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.show()
    """

    
    kern = kernel.Kernel()
    kern = kern.Kg2(7, 7, sigma=5.75, muX=0.0, muY=0.0)
    kern /= np.sum(kern)        # normalize volume

    plt.imshow(np.real(IFFT(FFT(kern)*FFT(image[:kern.shape[0], :kern.shape[1]]))), \
        interpolation='none', cmap='gray')
    plt.show()

    #conv = convolve(image[:2*kern.shape[0],:5*kern.shape[1]], kern)
    conv = convolve(image, kern)

    plt.imshow(conv.OaAconvolve(), interpolation='none', cmap='gray')
    plt.show()