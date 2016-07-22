#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" give lists entry-wise products, powers, division, addition and
subtraction by other lists, as though they were vectors. This code
is probably way over the top, but it's simple and it works """

from warnings import warn

class entrylist(list):
    """ An iterable with useful attributes from lists and entrylists.

    Example usage:

    >>> from numpy.random import randn
    >>> entrylist(randn(5,6).shape)
    [5, 6]
    >>> X = entrylist(randn(5,6).shape)
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
    >>> X
    [5, 6]
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
            return entrylist(i // other for i in self)
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return entrylist(i // j for i, j in zip(self, other))
        else:
            raise TypeError

    def __div__(self, other):
        if (isinstance(other, (int, float))):
            return entrylist(i / other for i in self)
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return entrylist(i / j for i, j in zip(self, other))
        else:
            raise TypeError

    def __mul__(self, other):
        if (isinstance(other, int)):
            return entrylist(i * other for i in self)
        elif (isinstance(other, (list, entrylist))):
            # multiply entry-wise
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return entrylist(i * j for i, j in zip(self, other))
        else:
            raise TypeError

    def __pow__(self, other):
        if (isinstance(other, (int, float))):
            return entrylist(i ** other for i in self)
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return entrylist(i ** j for i, j in zip(self, other))
        else:
            raise TypeError

    def __add__(self, other):
        if (isinstance(other, (int, float))):
            return entrylist(i + other for i in self)
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return entrylist(i + j for i, j in zip(self, other))
        else:
            raise TypeError

    def __sub__(self, other):
        return self.__add__(-other)

    def __le__(self, other):
        if (isinstance(other, (int, float))):
            return all(entrylist(i <= other for i in self))
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return all(entrylist(i <= j for i, j in zip(self, other)))
        else:
            raise TypeError

    def __ge__(self, other):
        if (isinstance(other, (int, float))):
            return all(entrylist(i >= other for i in self))
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return all(entrylist(i >= j for i, j in zip(self, other)))
        else:
            raise TypeError

    def __eq__(self, other):
        if (isinstance(other, (int, float))):
            return all(entrylist(i == other for i in self))
        elif (isinstance(other, (list, entrylist))):
            if (other.__len__() != self.__len__()):
                warn('len(other) != len(self): Possible data loss')
            return all(entrylist(i == j for i, j in zip(self, other)))
        else:
            raise TypeError

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return not self.__ge__(other)

    def __gt__(self, other):
        return not self.__le__(other)