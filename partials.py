#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import numpy as np
from numpy.linalg import norm

### Special case - 2 Dimensional

def m2(X, theta):
    """ Moffat distribution """
    return theta[0]*(theta[2] - 1.0)/(np.pi*theta[1]**2.0)*(1.0 + ((X[0] - \
        theta[3])**2.0+(X[1] - theta[4])**2.0)/(theta[1]**2.0))**(-theta[2])

def g2(X, theta):
    """ Gaussian Distribution """
    return theta[0]/(2.0*np.pi*theta[1]**2.0)*np.exp(-((X[0]-theta[2])**2.0\
        +(X[1]-theta[3])**2.0)/(2.0*theta[1]**2.0))

### N-Dimensional Distributions

def mn(X, theta):
    """ n-dimensional Moffat distribution """
    return theta[0]*(theta[2] - 1.0)/(np.pi*theta[1]**2.0)\
        *(1.0 + sum((X[i] - theta[3:][i])**2.0 for i in \
        xrange(len(X)))/(theta[1]**2.0))**(-theta[2])

def gn(X, theta):
    """ n-dimensional Gaussian distribution """
    return theta[0]/(((2.0*np.pi)**(len(X) / 2.0))*(theta[1]**len(X)))\
        *np.exp(-(1.0 / (2.0*(theta[1]**2.0)))\
        *sum((X[i]-theta[2:][i])**2.0 for i in xrange(len(X))))

### Gaussian

def PartialAg2(X, theta):
    return g2(X, theta) / theta[0]

def PartialSigmaXg2(X, theta):
    if (type(X).__module__ != np.__name__):
        X = np.array(X)
    if (type(theta).__module__ != np.__name__):
        theta = np.array(theta)
    return g2(X, theta)*((norm(X - theta[2:])**2.0)/(theta[1]**3.0)-2.0/\
        theta[1])

def PartialMuXg2(X, theta, NAXIS=0):
    if (theta[1] < 0.0):
        # σ must be >= 0.0
        theta[1] = 0.0
    return (g2(X, theta)*(X[NAXIS] - theta[NAXIS + 2])) / (theta[1]**2.0)

### Moffat

def PartialAm2(X, theta):
    return m2(X, theta) / theta[0]

def PartialAlpham2(X, theta):
    if (type(X).__module__ != np.__name__):
        X = np.array(X)
    if (type(theta).__module__ != np.__name__):
        theta = np.array(theta)

    return (2.0*m2(X, theta)/theta[1])*((theta[2]*norm(X - theta[3:])**2.0)/\
        (theta[1]**2.0 + norm(X - theta[3:])**2.0) - 1.0)

def PartialBetam2(X, theta):
    if (type(X).__module__ != np.__name__):
        X = np.array(X)
    if (type(theta).__module__ != np.__name__):
        theta = np.array(theta)
    
    return m2(X, theta)*(log((theta[1]**2.0) / \
        (theta[1]**2.0 + norm(X - theta[3:])**2.0)) + 1.0 / (theta[2] - 1.0))

def PartialMuXm2(X, theta, NAXIS=0):
    if (type(X).__module__ != np.__name__):
        X = np.array(X)
    if (type(theta).__module__ != np.__name__):
        theta = np.array(theta)
    
    return (2.0*m2(X, theta)*theta[2]*(X[NAXIS] - theta[3 + NAXIS]))/\
        (theta[1]**2.0 + norm(X - theta[3:])**2.0)
