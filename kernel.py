#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.integrate as integrate
import numpy as np
import partials

class Kernel(object):
    """ creates kernels """
    def __init__(self, random_noise=False, noise_level=0.001):
        self.random_noise = random_noise
        if (random_noise):
            self.noise_level = noise_level

    def Kg2(self, X, Y, sigma, muX=0.0, muY=0.0):
        """ return a 2 dimensional Gaussian kernel """
        kernel = np.zeros([X, Y])
        theta = [1.0, sigma, muX, muY]
        for i in xrange(X):
            for j in xrange(Y):
                kernel[i][j] = integrate.dblquad(lambda x, y: \
                                partials.g2([x + float(i) - (X-1.0)/2.0, \
                                y + float(j) - (Y-1.0)/2.0], theta), \
                                -0.5, 0.5, lambda y: -0.5, lambda y: 0.5)[0]

        if (self.random_noise):
            kernel = kernel + np.abs(np.random.randn(X, Y))*self.noise_level
            maximum = np.max(kernel)
            minimum = np.min(kernel)
            return map(lambda arr: (arr-minimum)/(maximum-minimum), kernel)
        else:
            maximum = np.max(kernel)
            minimum = np.min(kernel)
            print maximum, minimum
            return map(lambda arr: (arr-minimum)/(maximum-minimum), kernel)

    @staticmethod
    def FWHMg2(theta):
        """ get the FWHM of the Gaussian distribution """
        return 2*theta[3]*sqrt(log(1.0 / \
            ((pi**2.0)*(theta[3])**4.0)))
    
    def Km2(self, X, Y, alpha, beta, muX=0.0, muY=0.0):
        """ return a 2 dimensional Moffat kernel """
        kernel = np.zeros([X, Y])
        theta = [1.0, alpha, beta, muX, muY]
        for i in xrange(X):
            for j in xrange(Y):
                kernel[i][j] = integrate.dblquad(lambda x, y: \
                                partials.m2([x+ float(i) - (X-1.0)/2.0, \
                                y + float(j) - (Y-1.0)/2.0], theta), \
                                -0.5, 0.5, lambda y: -0.5, lambda y: 0.5)[0]

        if (self.random_noise):
            kernel = kernel + np.abs(np.random.randn(X, Y))*self.noise_level
            maximum = np.max(kernel)
            minimum = np.min(kernel)
            return map(lambda arr: (arr-minimum)/(maximum-minimum), kernel)
        else:
            maximum = np.max(kernel)
            minimum = np.min(kernel)
            return map(lambda arr: (arr-minimum)/(maximum-minimum), kernel)

    @staticmethod
    def FWHMm2(theta):
        """ get the FWHM of the Moffat distribution """
        return 2*theta[0]*sqrt(pow(2, 1 / theta[1]) - 1.0)
