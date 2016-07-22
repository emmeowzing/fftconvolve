#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Quick implementation of several convolution algorithms to compare 
times.
"""

import numpy as np
import _kernel
from tqdm import trange, tqdm
from numpy.fft import fft2 as FFT, ifft2 as iFFT
from scipy.ndimage.filters import convolve

from PIL import Image
from scipy.misc import imsave
from time import time, sleep


__author__ = "Brandon Doyle"
__email__ = "bjd2385@aperiodicity.com"


class convolve(object):
    """ contains methods to convolve two images """
    def __init__(self, image_array, kernel):
        self.array = image_array
        self.kernel = kernel

        # Store these values as they will be accessed a _lot_
        self.__rangeKX_ = self.kernel.shape[0]
        self.__rangeKY_ = self.kernel.shape[1]
        
        # pad array for convolution
        self.__offsetX_ = self.__rangeKX_ // 2
        self.__offsetY_ = self.__rangeKY_ // 2
     
        self.array = np.lib.pad(self.array,      \
            [(self.__offsetX_, self.__offsetX_), \
             (self.__offsetY_, self.__offsetY_)],\
             mode='constant', constant_values=0)

        self.__rangeX_ = self.array.shape[0]
        self.__rangeY_ = self.array.shape[1]

        # to be returned instead of the originals
        self.__arr_ = np.zeros([self.__rangeX_, self.__rangeY_])

    def spaceConv(self):
        """ normal convolution, O(N^2*n^2). This is usually too slow """

        # this is the O(N^2) part of this algorithm
        for i in trange(self.__offsetX_, \
                self.__rangeX_ - self.__rangeKX_):
            for j in xrange(self.__offsetY_, \
                    self.__rangeY_ - self.__rangeKY_):
                # Now the O(n^2) portion
                total = 0.0
                for k in xrange(self.__rangeKX_):
                    for t in xrange(self.__rangeKY_):
                        total += \
                  self.kernel[k][t] * self.array[i+k][j+t]

                # Update entry in self.__arr_, which is to be returned
                # http://stackoverflow.com/a/38320467/3928184
                self.__arr_[i][j] = total

        return self.__arr_[self.__offsetX_\
                           :self.__rangeX_ - self.__offsetX_,\
                           self.__offsetY_\
                           :self.__rangeY_ - self.__offsetY_]

    def spaceConvDot(self):
        """ Exactly the same as the former method, just contains a 
        nested function so the dot product appears more obvious """

        def dot(ind, jnd):
            """ perform a simple 'dot product' between the 2 
            dimensional image subsets. """
            total = 0.0

            # This is the O(n^2) part of the algorithm
            for k in xrange(self.__rangeKX_):
                for t in xrange(self.__rangeKY_):
                    total += \
              self.kernel[k][t] * self.array[k + ind, t + jnd]
            return total
     
        # this is the O(N^2) part of the algorithm
        for i in trange(self.__offsetX_, \
                self.__rangeX_ - self.__rangeKX_):
            for j in xrange(self.__offsetY_, \
                    self.__rangeY_ - self.__rangeKY_):
                self.__arr_[i][j] = dot(i, j)
     
        return self.__arr_[self.__offsetX_\
                           :self.__rangeX_ - self.__offsetX_,\
                           self.__offsetY_\
                           :self.__rangeY_ - self.__offsetY_]

    @classmethod
    def InvertKernel(kernel):
        """ Invert a kernel for an example """

        X, Y = kernel.shape
        # thanks to http://stackoverflow.com/a/38384551/3928184!
        new_kernel = np.full_like(kernel, 0)

        for i in xrange(X):
            for j in xrange(Y):
                n_i = (i + X // 2) % X
                n_j = (j + Y // 2) % Y
                new_kernel[n_i, n_j] = kernel[i, j]

        return new_kernel

    @staticmethod
    def Interpolate(kernel, unit=1):
        """ Interpolate a kernel a single unit smaller or larger """


    def OAconv(self):
        """ faster convolution algorithm, O(N^2*log(n)). """

        # solve for the total padding along each axis
        diffX = (self.__rangeKX_ - self.__rangeX_ +  \
                 self.__rangeKX_ * (self.__rangeX_ //\
                 self.__rangeKX_)) % self.__rangeKX_
        
        diffY = (self.__rangeKY_ - self.__rangeY_ +  \
                 self.__rangeKY_ * (self.__rangeY_ //\
                 self.__rangeKY_)) % self.__rangeKY_

        # padding on each side, i.e. left, right, top and bottom; 
        # centered as well as possible
        right = diffX // 2
        left = diffX - right
        bottom = diffY // 2
        top = diffY - bottom

        # pad the array
        self.array = np.lib.pad(self.array,                 \
                            ((left, right), (top, bottom)), \
                           mode='constant', constant_values=0)

        divX = self.array.shape[0] / float(self.__rangeKX_)
        divY = self.array.shape[1] / float(self.__rangeKY_)

        # Let's just make sure...
        if (divX % 1.0 or divY % 1.0):
            raise ValueError('Image not partitionable')
        else:
            divX = int(divX)
            divY = int(divY)

        # a list of tuples to partition the array by
        subsets = [(i*self.__rangeKX_, (i + 1)*self.__rangeKX_,\
                    j*self.__rangeKY_, (j + 1)*self.__rangeKY_)\
                        for i in xrange(divX)                  \
                        for j in xrange(divY)]

        # padding for individual blocks in the subsets list
        padX = self.__rangeKX_ // 2
        padY = self.__rangeKY_ // 2

        self.__arr_ = np.lib.pad(self.__arr_,              \
                            ((left + padX, right + padX),  \
                             (top + padY, bottom + padY)), \
                           mode='constant', constant_values=0)

        kernel = np.pad(self.kernel,                  \
                        [(padX, padX), (padY, padY)], \
                      mode='constant', constant_values=0)

        # thanks to http://stackoverflow.com/a/38384551/3928184!
        # Invert the kernel
        X, Y = kernel.shape
        new_kernel = np.full_like(kernel, 0)

        for i in xrange(X):
            for j in xrange(Y):
                n_i = (i + X // 2) % X
                n_j = (j + Y // 2) % Y
                new_kernel[n_i, n_j] = kernel[i, j]

        # We only need to do this once
        transf_kernel = FFT(new_kernel)

        # transform each partition and OA on conv_image
        for tup in tqdm(subsets):
            # slice and pad the array subset
            subset = self.array[tup[0]:tup[1], tup[2]:tup[3]]

            subset = np.lib.pad(subset,      \
                [(padY, padY), (padX, padX)],\
                mode='constant', constant_values=0)

            transf_subset = FFT(subset)

            # multiply the two arrays entrywise
            space = iFFT(transf_subset * transf_kernel).real

            # overlap with indices and add them together
            self.__arr_[tup[0]:tup[1] + 2 * padX, \
                        tup[2]:tup[3] + 2 * padY] += space

        # crop image and get it back, convolved
        return self.__arr_[self.__offsetX_ + padX + left   \
                           :padX + left + self.__rangeX_   \
                              - self.__offsetX_,           \
                           self.__offsetY_ + padY + bottom \
                           :padY + bottom + self.__rangeY_ \
                              - self.__offsetY_]

    def builtin(self):
        """ Convolves using SciPy's convolution function - extremely
        fast """
        from scipy.ndimage.filters import convolve
        return convolve(self.array, self.kernel)