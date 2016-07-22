#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def clock(*axes):
    """ A quick generator for working with iterations over N-dimensional
    arrays """
    axes = np.array(axes)
    # increases continually until axes.prod()
    iteration = 0
    dimension = len(axes)
    maxiterations = axes.prod()
    
    if not (dimension >= 1):
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