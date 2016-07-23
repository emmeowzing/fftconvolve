#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" give lists entry-wise products, powers, division, addition and
subtraction by other lists, as though they were vectors. This code
is probably way over the top, but it's simple and it works """

from warnings import warn
from clock import clock

class operalist(list):
    """ An iterable with useful attributes from lists and operalists.

    Example usage:

    >>> from numpy.random import randn
    >>> X = operalist(randn(5,6).shape)
    >>> X
    [5, 6]
    >>> X // 2
    [2, 3]
    >>> X / 2
    [2, 3]
    >>> X * 2
    [10, 12]
    >>> X + 2
    [7, 8]
    >>> X - 2
    [3, 4]
    >>> X > 2
    True
    >>> X < 2
    False
    >>> X <= 2
    False
    >>> X >= 2
    True
    """
    def __floordiv__(self, other):
        if (isinstance(other, (int, float))):
            return operalist(i // other for i in self)
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            
            return operalist(i // j for i, j in zip(self, other))
        
        else:
            raise TypeError

    def __div__(self, other):
        if (isinstance(other, (int, float))):
            return operalist(i / other for i in self)
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
        
            return operalist(i / j for i, j in zip(self, other))
        
        else:
            raise TypeError

    def __mul__(self, other):
        if (isinstance(other, int)):
            return operalist(i * other for i in self)
        
        elif (isinstance(other, (list, operalist))):
        
            # multiply entry-wise
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
        
            return operalist(i * j for i, j in zip(self, other))
        
        else:
            raise TypeError

    def __pow__(self, other):
        if (isinstance(other, (int, float))):
            return operalist(i ** other for i in self)
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
        
            return operalist(i ** j for i, j in zip(self, other))
        
        else:
            raise TypeError

    def __add__(self, other):
        if (isinstance(other, (int, float))):
            return operalist(i + other for i in self)
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
        
            return operalist(i + j for i, j in zip(self, other))
        
        else:
            raise TypeError

    def __sub__(self, other):
        return self.__add__(-other)

    def __le__(self, other):
        if (isinstance(other, (int, float))):
            return all(operalist(i <= other for i in self))

        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            
            return all(operalist(i <= j for i, j in zip(self, other)))
        
        else:
            raise TypeError

    def __ge__(self, other):
        if (isinstance(other, (int, float))):
            return all(operalist(i >= other for i in self))
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
        
            return all(operalist(i >= j for i, j in zip(self, other)))
        
        else:
            raise TypeError

    def __eq__(self, other):
        if (isinstance(other, (int, float))):
            return all(operalist(i == other for i in self))
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
        
            return all(operalist(i == j for i, j in zip(self, other)))
        
        else:
            raise TypeError

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return not self.__ge__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __mod__(self, other):
        if (isinstance(other, (int, float))):
            return operalist(i % other for i in self)
        
        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            
            return operalist(i % j for i, j in zip(self, other))

        else:
            raise TypeError

    def __divmod__(self, other):
        if (isinstance(other, (int, float))):
            return operalist((i / other, i % other) for i in self)

        elif (isinstance(other, (list, operalist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')

            return operalist(operalist([i / j, i % j]) \
                for i, j in zip(self, other))

        else:
            raise TypeError

    def __float__(self):
        return operalist(float(i) for i in self)

## Following needs to be worked out

class multioperalist(operalist):
    """ Build on operalists with multidimensional operalists """
    pass