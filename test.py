#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A quick example :P """

from fftconv import *
from scipy.misc import imsave, imread
from math import log10
import _kernel
import numpy as np
import matplotlib.pyplot as plt

from time import time

def main():
    ## Initialize a kernel
    times1, times2, times3, times4 = ([],) * 4
    domain = [3, 7, 2]

    ## open an image
    image = np.rot90(np.rot90(np.rot90(imread('spider.jpg').T[0])))

    for i in xrange(*domain):
        kern = _kernel.Kernel()
        kern = kern.Kg2(i, i, sigma=i * 0.5, muX=0.0, muY=0.0)
        kern /= np.sum(kern)    # normalize volume

        ## Convolve using some method
        conv = convolve(image, kern)

        # First method
        _start = time()
        X = conv.spaceConv2()
        times1.append(log10(time() - _start))
        #plt.imshow(X, interpolation='none', cmap='gray')
        #plt.show()

        # Second method
        _start = time()
        X = conv.spaceConvDot2()
        times2.append(log10(time() - _start))
        #plt.imshow(X, interpolation='none', cmap='gray')
        #plt.show()

        # Third (NumPy)
        _start = time()
        X = conv.spaceConvNumPy2()
        times3.append(log10(time() - _start))
        #plt.imshow(X, interpolation='none', cmap='gray')
        #plt.show()

        # Fourth (Numba)
        _start = time()
        X = conv.spaceConvNumba2()
        times4.append(log10(time() - _start))
        #plt.imshow(X, interpolation='none', cmap='gray')
        #plt.show()
        print

    plt.plot(times1, label='Pure')
    plt.plot(times2, label='Nested Dot')
    plt.plot(times3, label='NumPy')
    plt.plot(times4, label='Jit')

    plt.xlim([-0.5, len(times1) + 0.5])

    # modify xticks to represent data
    plt.xticks(range(len(times1)), 
        [str(i) for i in range(*domain)]
    )

    plt.legend(fontsize=12, loc='upper left')
    plt.xlabel(r'Kernel Side Length', fontsize=12)
    plt.ylabel(r'Time (s)', fontsize=12)
    plt.title('Array Size vs. Time for Numba, NumPy and Pure Python', fontsize=12)
    plt.show()

if __name__ == '__main__':
    main()
