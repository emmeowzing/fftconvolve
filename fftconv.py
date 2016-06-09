#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple implementation of the fastest FFT convolution algorithm, as well as
several others.

O(N^2*log(n)) time, where N^2 is the size of the image (we can never get below 
this of course, but we may _approach_ it).
"""

import numpy as np
from numpy.fft import fft2, ifft2
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

        def pad(array, left, right=None, top=None, bottom=None):
            """ pad an array with values """
            if left is not None:
                return np.pad(array, [(left, left), (left, left)], \
                    mode='constant', constant_values=0)
            elif left is not None and top is not None:
                return np.pad(array, [(left, right), (left, right)], \
                    mode='constant', constant_values=0)
            else:
                return np.pad(array, [(left, right), (top, bottom)], \
                    mode='constant', constant_values=0)
        
        def partition():
            """ solve for the divisibility and pad """

            # solve for the total padding along each axis
            diffX = (self.__rangeKX_ -                      \
                        (self.__rangeX_ - self.__rangeKX_*( \
                        self.__rangeX_ // self.__rangeKX_)))\
                        % self.__rangeKX_
            
            diffY = (self.__rangeKY_ -                      \
                        (self.__rangeY_ - self.__rangeKY_*( \
                        self.__rangeY_ // self.__rangeKY_)))\
                        % self.__rangeKY_

            # each side, i.e. left, right, top and bottom
            right = diffX // 2
            left = diffX - right
            bottom = diffY // 2
            top = diffY - bottom

            # pad the array
            self.array = pad(self.array, left, right, top, bottom)
            
            # return a list of tuples to partition the array
            return [(i*self.__rangeKX_, (i + 1)*self.__rangeKX_,              \
                     j*self.__rangeKY_, (j + 1)*self.__rangeKY_)              \
                     for i in xrange(self.array.shape[0] // self.__rangeKX_)  \
                     for j in xrange(self.array.shape[1] // self.__rangeKY_)],\
                     left, right, top, bottom
        
        subsets, left, right, top, bottom  = partition()
        
        # set up OaA method        
        padX = self.__rangeKX_ // 2
        padY = self.__rangeKY_ // 2

        transformed_kernel = fft2(pad(self.kernel, padX, padX, padY, padY))

        # create a blank array of the same size
        convolved_image = np.zeros([self.array.shape[0]+left+right+2*padX, \
                                    self.array.shape[1]+bottom+top+2*padY])

        # transform each partition and OaA on the convolved_image
        for tup in tqdm(subsets):
            # transform
            transformed_image_subset = \
                fft2(pad(self.array[tup[0]:tup[1], tup[2]:tup[3]], \
                     padX, padX, padY, padY))
            
            # multiply the two arrays together and take the IFFT
            space = np.real(ifft2(transformed_kernel*transformed_image_subset))

            convolved_image[tup[0]:tup[1]+padX,tup[2]-padY:tup[3]+padY]+=\
                space[0:2*padX + self.__rangeKX_, 0:2*padY + self.__rangeKX_]

        # crop image and get it back, convolved
        return convolved_image[padX:padX + self.__rangeX_, \
                               padY:padY + self.__rangeY_]

if __name__ == '__main__':
    try:
        import pyplot as plt
    except ImportError:
        import matplotlib.pyplot as plt

    image = np.array(\
        Image.open('/home/brandon/Pictures/lune_2010-09-29_06-09-07-s1.jpg'))

    image = image.T[0]

    kern = kernel.Kernel()
    kern = kern.Kg2(7, 7, sigma=1.5, muX=0.0, muY=0.0)
    kern /= np.sum(kern)        # normalize volume
    plt.imshow(kern, interpolation='none', cmap='gist_heat')
    plt.colorbar()
    plt.show()

    conv = convolve(image, kern)

    plt.imshow(conv.OaAconvolve(), interpolation='none', cmap='gray')
    plt.show()