#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Quick implementation of several convolution algorithms to compare 
times. I don't think there's anything incredibly new in this code, I've
just written it to better-understand Python, OOP, convolution
algorithms and (eventually) higher-dimensional programming.
"""

import numpy as np
from tqdm import trange, tqdm
from numpy.fft import fft2 as FFT, ifft2 as iFFT
from numpy.fft import rfft2 as rFFT, irfft2 as irFFT
from numpy.fft import fftn as FFTN, ifftn as iFFTN
from numba import jit
from multiprocessing.dummy import Pool as ThreadPool
from psutil import cpu_count


__author__ = "Brandon Doyle"
__email__ = "bjd2385@aperiodicity.com"


class convolve(object):
    """ contains methods to convolve two images """
    def __init__(self, image_array, kernel):
        self.array = image_array
        self.kernel = kernel

        self.__rangeX_ , self.__rangeY_  = image_array.shape
        self.__rangeKX_, self.__rangeKY_ = kernel.shape

        # to be returned instead of the original
        self.__arr_ = np.zeros(image_array.shape)
        
        # pad array for convolution
        self.__offsetX_ = self.__rangeKX_ // 2
        self.__offsetY_ = self.__rangeKY_ // 2
        
        self.array = np.lib.pad(self.array
            , [(self.__offsetY_, self.__offsetY_),
               (self.__offsetX_, self.__offsetX_)]
            , mode='constant'
            , constant_values=0
        )

    ### There are 4 different spacial convolution algorithms

    def spaceConv2(self):
        """ normal convolution, O(N^2*n^2). This is usually too slow """

        # this is the O(N^2) part of this algorithm
        for i in trange(self.__rangeX_):
            for j in xrange(self.__rangeY_):
                # Now the O(n^2) portion
                total = 0.0
                for k in xrange(self.__rangeKX_):
                    for t in xrange(self.__rangeKY_):
                        total += self.kernel[k][t] * self.array[i+k, j+t]

                # Update entry in self.__arr_, which is to be returned
                # http://stackoverflow.com/a/38320467/3928184
                self.__arr_[i, j] = total

        return self.__arr_

    def spaceConvDot2(self):
        """ Exactly the same as the former method, just contains a 
        nested function so the dot product appears more obvious """

        def dot(ind, jnd):
            """ perform a simple 'dot product' between the 2 
            dimensional image subsets. """
            total = 0.0

            # This is the O(n^2) part of the algorithm
            for k in xrange(self.__rangeKX_):
                for t in xrange(self.__rangeKY_):
                    total += self.kernel[k][t] * self.array[k+ind, t+jnd]
            return total
     
        # this is the O(N^2) part of the algorithm
        for i in trange(self.__rangeX_):
            for j in xrange(self.__rangeY_):
                self.__arr_[i, j] = dot(i, j)
     
        return self.__arr_

    ## Following are speedups using either Numba or NumPy

    def spaceConvNumPy2(self):
        # this is the O(N^2) part of the algorithm

        @checkarrays
        def dotNumPy(subarray):
            return np.sum(self.kernel * subarray)

        for i in trange(self.__rangeX_):
            for j in xrange(self.__rangeY_):
                self.__arr_[i, j] = dotNumPy(\
                    self.array[i:i + self.__rangeKX_,
                               j:j + self.__rangeKY_]
                )

        return self.__arr_

    def spaceConvNumba2(self):
        """ Exactly the same as the former method, just contains a 
        nested function so the dot product appears more obvious """ 

        @checkarrays
        @jit
        def dotJit(subarray, kernel):
            """ perform a simple 'dot product' between the 2 dimensional 
            image subsets. 
            """
            total = 0.0
            # This is the O(n^2) part of the algorithm
            for i in xrange(subarray.shape[0]):
                for j in xrange(subarray.shape[1]):
                    total += subarray[i][j] * kernel[i][j]
            return total
     
        # this is the O(N^2) part of the algorithm
        for i in trange(self.__rangeX_):
            for j in xrange(self.__rangeY_):
                # dotJit is located outside the class :P
                self.__arr_[i, j] = dotJit(\
                    self.array[i:i+self.__rangeKX_,
                               j:j+self.__rangeKY_]
                    , self.kernel
                )
     
        return self.__arr_

    def spaceConvNumbaThreadedOuter2(self):
        """ `Block` threading example """

        def partitioner(shape, n_cores):
            """ Partition an array for multithreading as evenly as possible
            """
            if (n_cores == 1):    # 0 < n_cores < 2
                return shape, n_cores
            elif (n_cores < 1):
                raise ValueError(\
            'partitioner expected a positive number of cores, got %d'\
                    % n_cores
                )
            elif (n_cores % 2):
                raise ValueError(\
               'partitioner expected an even number of cores, got %d'\
                    % n_cores
                )

            # partitioning on the larger of the two dirs if n_cores is 
            # > shape to maximize the number of threads that can be used
            if ((n_cores, n_cores) > tuple(shape)):
                n_cores = max(shape)
                axis = shape.index(n_cores)
            else:
                # just partition vertically (could be either axis, really)
                axis = 0

            step = 1.0 / n_cores
            partition = []
            for i in xrange(1, n_cores + 1):
                partition.append(int(i * step * shape[axis]))

            if (axis == 0):
                result = zip(partition,\
                    [shape[(axis + 1) % 2]]*len(partition)
                )
            elif (axis == 1):
                result = zip([shape[(axis + 1) % 2]]*len(partition),\
                    partition
                )

            result.append((0, 0))
            result = sorted(result)
            new_result = []
            for i in xrange(len(result) - 1):
                if not (i):
                    new_result.append((result[i], result[i+1])) 
                else:
                    new_result.append(
                        ((result[i][0], result[i-1][0]),
                        (result[i+1]))
                    )

            return sorted(new_result), n_cores

        @checkarrays
        @jit
        def dotJit(subarray, kernel):
            total = 0.0
            for i in xrange(subarray.shape[0]):
                for j in xrange(subarray.shape[1]):
                    total += subarray[i][j] * kernel[i][j]
            return total

        def outer(subset):
            for i in xrange(subset[0][0], subset[1][0]):
                for j in xrange(subset[0][1], subset[1][1]):
                    self.__arr_[i, j] = dotJit(\
                        self.array[i:i+self.__rangeKX_,
                                   j:j+self.__rangeKY_]
                        , self.kernel
                    )

        cores = cpu_count()
        cores -= 1 if (cores % 2 == 1 and cores > 1) else 0

        # Get partitioning indices and the usable number of cores
        shape = [self.__rangeX_, self.__rangeY_]
        partitions, usable_cores = partitioner(shape, cores)
        
        # Map partitions to threads; works in-place, no return
        pool = ThreadPool(usable_cores)
        pool.map(outer, partitions)
        pool.close()
        pool.join()

        return self.__arr_

    '''
    def spaceConvNumbaThreadedInner2(self):
        """ `Speckled` threading example, in the event that I have some
        spare time 
        """
        
        @checkarrays
        @jit
        def dotJit(subarray, kernel):
            total = 0.0
            # This is the O(n^2) part of the algorithm
            for i in xrange(subarray.shape[0]):
                for j in xrange(subarray.shape[1]):
                    total += subarray[i][j] * kernel[i][j]
            return total

        # Set up cores
        cores = cpu_count()
        cores -= 1 if (cores % 2 == 1 and cores > 1) else 0
        pool = ThreadPool(cores)

        if (self.__rangeX_ % cores):
            # Width of image is not evenly divisible by # of cores
            last = self.__rangeX_ // cores + 1
            for i in xrange(last):
                if (i is last):
                    # Split vertically to catch an extra few ms
                    if (self.__rangeY_ % 2):
                        for j in xrange(self.__rangeY_ // cores + 1):
                            for k in xrange(cores):
                                results = pool.map(dotJit, )

                    else:
                        for j in xrange(self.__range_Y // cores):
                            for k in xrange(cores):
                                results = pool.map(dotJit, )

                else:
                    # Two
                    for j in xrange(self.__rangeY_):
                        for k in xrange(cores):
                            results = pool.map(dotJit, )

        else:
            # width of image is divisible by # of cores available
            for i in xrange(self.__rangeX_ // cores):
                for j in xrange(self.__rangeY_):
                    for k in xrange(cores):
                        results =  pool.map(dotJit, )

                    for k in xrange(cores):
                        # assign values
                        self.__arr_[i+k, j] = results[k]

        # Now map a separate thread to process each partition
        pool.close()
        pool.join()

        return self.__arr_

    '''

    ### End of spacial convolution algorithms

    @staticmethod
    def InvertKernel2(kernel):
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

    ### Start of special convolution algorithms using the FFT

    def FFTconv2(self):
        """ FFT convolution, not quite OAconv, but its all in NumPy """
        # just overwrite this array since it's already allocated
        self.__arr_ = irFFT(rFFT(self.array) * rFFT(self.kernel, \
                                         self.array.shape))

        return self.__arr_

    def OAconv2(self):
        """ A threaded version of the former algorithm """
        self.__rangePX_, self.__rangePY_ = self.array.shape

        diffX = (self.__rangeKX_ - self.__rangePX_ +  \
                 self.__rangeKX_ * (self.__rangePX_ //\
                 self.__rangeKX_)) % self.__rangeKX_
        diffY = (self.__rangeKY_ - self.__rangePY_ +  \
                 self.__rangeKY_ * (self.__rangePY_ //\
                 self.__rangeKY_)) % self.__rangeKY_

        # padding on each side, i.e. left, right, top and bottom; 
        # centered as well as possible
        right = diffX // 2
        left = diffX - right
        bottom = diffY // 2
        top = diffY - bottom

        # pad the array
        self.array = np.lib.pad(self.array
            , ((left, right), (top, bottom))
            , mode='constant'
            , constant_values=0
        )

        divX = int(self.array.shape[0] // self.__rangeKX_)
        divY = int(self.array.shape[1] // self.__rangeKY_)

        # a list of tuples to partition the array by
        subsets = [(i*self.__rangeKX_, (i + 1)*self.__rangeKX_,\
                    j*self.__rangeKY_, (j + 1)*self.__rangeKY_)\
                   for i in xrange(divX)\
                   for j in xrange(divY)]

        # padding for individual blocks in the subsets list
        padX = self.__rangeKX_ // 2
        padY = self.__rangeKY_ // 2

        # Add. padding for __arr_ so it can store the results
        self.__arr_ = np.lib.pad(self.__arr_
            , ((left+padX+self.__offsetX_, right+padX+self.__offsetX_), 
               (top+padY+self.__offsetY_, bottom+padY+self.__offsetY_))
            , mode='constant', constant_values=0
        )

        kernel = np.pad(self.kernel
            , [(padX, padX), (padY, padY)]
            , mode='constant'
            , constant_values=0
        )

        # thanks to http://stackoverflow.com/a/38384551/3928184!
        # Invert the kernel
        new_kernel = self.InvertKernel2(kernel)
        transf_kernel = FFT(new_kernel)

        # transform each partition and OA on conv_image
        for tup in tqdm(subsets):
            # slice and pad the array subset
            subset = self.array[tup[0]:tup[1], tup[2]:tup[3]]
            subset = np.lib.pad(subset
                , [(padX, padX), (padY, padY)]
                , mode='constant'
                , constant_values=0
            )

            transf_subset = FFT(subset)

            # multiply the two arrays entrywise
            
            space = iFFT(transf_subset * transf_kernel).real
            
            # overlap with indices and add them together
            self.__arr_[tup[0]:tup[1] + 2 * padX,\
                        tup[2]:tup[3] + 2 * padY] += space

        # crop image and get it back, convolved
        return self.__arr_[
            padX+left+self.__offsetX_
           :padX+left+self.__offsetX_+self.__rangeX_,
            padY+bottom+self.__offsetY_
           :padY+bottom+self.__offsetY_+self.__rangeY_
        ]

    '''
    def OAconvThreaded2(self):
        """ faster convolution algorithm, O(N^2*log(n)). """
        # solve for the total padding along each axis
        self.__rangeX_, self.__rangeY_ = self.array.shape
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

        self.__arr_ = np.lib.pad(self.__arr_,             \
                            ((left + padX, right + padX), \
                             (top + padY, bottom + padY)),\
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
        return self.__arr_[padX + left:padX + left + self.__rangeX_,
                           padY + bottom:padY + bottom + self.__rangeY_]
        '''

def checkarrays(f):
    """ Similar to the @accepts decorator """
    def new_f(*args, **kwd):
        assert reduce(lambda x, y: x == y, map(np.shape, args))\
        , """Array and Subarray must have same dimensions,
          got %s and %s"""\
          .replace('          ', '') % (args[0].shape, args[1].shape,)
        return f(*args, **kwd)
    return new_f