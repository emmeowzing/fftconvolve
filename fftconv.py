#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Quick implementation of several convolution algorithms to compare 
times. I don't think there's anything incredibly new in this code, I've
just written it to better-understand Python, OOP, convolution
algorithms and higher-dimensional programming (eventually).
"""

import numpy as np
import _kernel
from tqdm import trange, tqdm
from numpy.fft import fft2 as FFT, ifft2 as iFFT
from numpy.fft import rfft2 as rFFT, irfft2 as irFFT
from numpy.fft import fftn as FFTN, ifftn as iFFTN


__author__ = "Brandon Doyle"
__email__ = "bjd2385@aperiodicity.com"


class convolve(object):
    """ contains methods to convolve two images """
    def __init__(self, image_array, kernel):
        self.array = image_array
        self.kernel = kernel

        self.dimA = self.dim(image_array)
        self.dimK = self.dim(kernel)

        if (self.dimA < self.dimK):
            raise IndexError("""Cannot convolve an image with
                a higher dimensional image kernel"""\
                .replace('                ', ''))

        elif (self.dimA == self.dimK):
            # This is the easiest route
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

        '''
        else:
            # Convolving an image with a kernel of lesser dimension
            self.__rangesA_ = operalist(self.array.shape)
            self.__rangesK_ = operalist(self.kernel.shape)

            # pad array for convolution using operalists
            self.__offsets_ = operalist(self.__rangesK_) // 2

            if not (all(self.__rangesK_[i] % self.__rangesK_[0] \
                for i in xrange(len(self.__rangesK_)))):
                # Means the kernel is not a hypercube
                self.__cube_ = 0
            else:
                self.__cube_ = 1
                
            self.array = np.lib.pad(self.array, )
                
            self.__arr_ = np.zeros(list(self.__rangesA_))
        '''

    @staticmethod
    def dim(array):
        """ Get the dimension of a NumPy array - this is just useful """
        return len(array.shape)

    def spaceConv2(self):
        """ normal convolution, O(N^2*n^2). This is usually too slow """
        if (self.dimA != 2 or self.dimK != 2):
            raise ConvolveDimError(\
                'Use the higher dimensional analogue')

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

    def spaceConvDot2(self):
        """ Exactly the same as the former method, just contains a 
        nested function so the dot product appears more obvious """

        if (self.dimA != 2 or self.dimK != 2):
            raise ConvolveDimError(\
                'Use the higher dimensional analogue')

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

    @staticmethod
    def InvertKernel2(kernel):
        """ Invert a kernel for an example """
        if (self.dim(kernel) > 2):
            raise ConvolveDimError(\
                'Use the higher dimensional analogue')

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
    def InterpK(kernel, unit=1):
        """ Interpolate a kernel a single unit smaller or larger.
        A destructive process as the inverse won't yield the same
        array. """



    def FFTconv2(self):
        """ FFT convolution, not quite OAconv, but its all in NumPy """
        if (self.dimA != 2 or self.dimK != 2):
            raise ConvolveDimError(\
                'Use the higher dimensional analogue')

        return irFFT(rFFT(self.image) * rFFT(self.kernel, \
                                             self.image.shape))

    def OAconv2(self):
        """ faster convolution algorithm, O(N^2*log(n)). """

        if (self.dim(self.kernel) % 2):
            self.InterpK(self.kernel, unit=1)
        

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
        for tup in subsets:
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

class ConvolveTypeError(Exception):
    pass

class ConvolveDimError(Exception):
    pass