#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Quick implementation of several convolution algorithms to compare 
times 
"""

import numpy as np
import _kernel
from tqdm import trange, tqdm
from PIL import Image
from scipy.misc import imsave
from time import time, sleep


__author__ = "Brandon Doyle"
__email__ = "bjd2385@aperiodicity.com"


class convolve(object):
    """ contains methods to convolve two images """
    def __init__(self, image_array, kernel, back_same_size=True):
        self.array = image_array
        self.kernel = kernel

        # Store these values as they will be accessed a _lot_
        self.__rangeKX_ = self.kernel.shape[0]
        self.__rangeKY_ = self.kernel.shape[1]

        if (back_same_size):
            # pad array for convolution
            self.__offsetX_ = self.__rangeKX_ // 2
            self.__offsetY_ = self.__rangeKY_ // 2
         
            self.array = np.lib.pad(self.array,      \
                [(self.__offsetY_, self.__offsetY_), \
                 (self.__offsetX_, self.__offsetX_)],\
                 mode='constant', constant_values=0)

            # Update these
            self.__rangeX_ = self.array.shape[0]
            self.__rangeY_ = self.array.shape[1]
        else:
            self.__rangeX_ = self.array.shape[0]
            self.__rangeY_ = self.array.shape[1]
            self.__offsetX_ = 0
            self.__offsetY_ = 0

        # to be returned instead of the originals
        self.__arr_ = np.zeros([self.__rangeX_, self.__rangeY_])

    def spaceConv(self):
        """ normal convolution, O(N^2*n^2). This is usually too slow """

        # this is the O(N^2) part of this algorithm
        for i in trange(self.__rangeX_):
            for j in xrange(self.__rangeY_):
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
        """ Exactly the same as the former method """

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
        for i in trange(self.__rangeX_):
            for j in xrange(self.__rangeY_):
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

    def OAconv(self):
        """ faster convolution algorithm, O(N^2*log(n)). """
        from numpy.fft import fft2 as FFT, ifft2 as iFFT

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
        if not (divX % 1.0 == 0.0 or divY % 1.0 == 0.0):
            raise ValueError('Image not partitionable (?)')
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


if __name__ == '__main__':
    try:
        import pyplot as plt
    except ImportError:
        import matplotlib.pyplot as plt

    image = np.array(Image.open(\
        '/home/brandon/Documents/fftconvolve/Nature.jpg'))

    image = np.rot90(np.flipud(np.fliplr(image.T[0])))

    '''
    times = []
    for i in range(4, 50, 2):
        kern = _kernel.Kernel()
        kern = kern.Kg2(i, i, sigma=1.5, muX=0.0, muY=0.0)
        kern /= np.sum(kern)        # normalize volume

        conv = convolve(image, kern)
    
        # Time the result of increasing kernel size
        _start = time()
        convolved = conv.OAconv()
        #   convolved = conv.builtin()
        _end = time()
        times.append(_end - _start)

    x = np.array(range(4, 50, 2))
    plt.plot(range(4, 50, 2), times)
    plt.title('Kernel Size vs. OAconv time', fontsize=12)
    plt.xlabel('Kernel Size (px)', fontsize=12)
    plt.ylabel('Time (s)', fontsize=12)
    plt.xticks(x, x)
    plt.show()
    '''

    kern = _kernel.Kernel()
    kern = kern.Kg2(10, 10, sigma=4.25, muX=0.0, muY=0.0)
    kern /= np.sum(kern)        # Normalize volume

    conv = convolve(image, kern)
    convolved = conv.OAconv()

    print image.shape, convolved.shape

    #conv = convolve(image[:2*kern.shape[0],:5*kern.shape[1]], kern)

    plt.imshow(convolved, interpolation='none', cmap='gray')
    plt.show()
    
    imsave('spider_oa1.png', convolved, format='png')