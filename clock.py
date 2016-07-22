#!/usr/bin/env python
# -*- coding: utf-8 -*-

def clock(*axes):
    """ A quick generator for working with iterations over N-dimensional
    arrays. Example usage:
    
    >>> from clock import clock
    >>> for i in clock(2, 3):
    ...     print i
    (0, 0)
    (0, 1)
    (0, 2)
    (1, 0)
    (1, 1)
    (1, 2)
    
    So that whilst iterating over NumPy arrays, we might try
    
    >>> from numpy.random import randn
    >>> from clock import clock
    >>> X = randn(2, 3)
    >>> for i in clock(*(X.shape)):
    ...     print X[i]
    ... 
    0.557724928504
    0.65652304739
    0.538328247903
    -0.722400267643
    1.3994330752
    0.142266875233
    """
    iteration = 0
    dimension = len(axes)
    maxiterations = __prod_(axes)
    
    if (dimension < 1):
        raise ValueError('Must submit valid number of axes')
    
    while (iteration < maxiterations):
        # Set up the inner loop
        coordinate = []
        ind = iteration

        # Build the coordinate
        for i in xrange(dimension):
            s = axes[dimension-i-1]
            g = ind % s
            ind /= s
            coordinate.append(g)

        iteration += 1
        yield tuple(reversed(coordinate))

def __prod_(vec):
    prod = 1
    for i in xrange(len(vec)):
        prod *= vec[i]
    return prod