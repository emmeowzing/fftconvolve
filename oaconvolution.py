#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-

""" A simple implementation of OaA convolution """

from PIL import Image
from typing import Union, Tuple
from types import FunctionType as Function
from functools import wraps
from glob import iglob
from scipy.integrate import dblquad
from math import pi
from numpy.fft import fft2 as FFT, ifft2 as iFFT
import numpy as np
import matplotlib.pyplot as plt

DIR = '/home/brandon/Documents/fftconvolve/*'
_either = any


def spacial_convolve(im: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    X, Y = im.shape
    kX, kY = kernel.shape
    edgeX, edgeY = kX // 2, kY // 2
    res = np.zeros(im.shape)
    im = np.lib.pad(im, ((edgeX,)*2, (edgeY,)*2), mode='constant')

    # Spacially loop over the array and compute the dot product at every point
    for i in range(X):
        for j in range(Y):
            res[i, j] = np.sum(np.multiply(im[i:i + kX, j:j + kY], kernel))

    return res


def scale(f: Function) -> Function:
    """ Decorator to conserve information """
    @wraps(f)
    def _new_f(*args, **kwargs) -> np.ndarray:
        arr = f(*args, **kwargs)
        # Scale array by volume (in this case \sum_{i,j}k_{i,j} in the discrete case
        # is the same as a double integral to find volume on
        # \iint_{[-3 * sigma, 3 * sigma]} in the continuous case/
        return arr / np.sum(arr)
    return _new_f


class Kernel:
    """ Generate different kernels for testing """
    def __init__(self, dims: Tuple[int, int]) -> None:

        # OaA convolution requires, in the way I've written it, a kernel with even
        # dimensions
        if _either(map(lambda x: x % 2, dims)):
            raise ValueError(
                'Kernel expected even dims, received %s' % (dims,)
            )

        self.dims = dims

    @scale
    def gaussian(self, sigma: float =1.0) -> np.ndarray:
        kernel = np.zeros(self.dims)
        unit_square = (-0.5, 0.5, lambda y: -0.5, lambda y: 0.5)

        x_shift = 1.0 if self.dims[0] % 2 else 0.5
        y_shift = 1.0 if self.dims[1] % 2 else 0.5

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                # integrate on a unit square centered at the origin as the
                # function moves about it
                kernel[i][j] = dblquad(
                    lambda x, y: 1 / (2 * pi * sigma ** 2) * np.exp(
                        - ((x + i - self.dims[0] // 2 + x_shift) / sigma) ** 2
                        - ((y + j - self.dims[1] // 2 + y_shift) / sigma) ** 2),
                    *unit_square
                )[0]

        return kernel

    def laplacian(self) -> np.ndarray:
        return np.array([[ 0, -1,  0],
                         [-1,  4, -1],
                         [ 0, -1,  0]]) / 4

    @scale
    def identity(self) -> np.ndarray:
        """ The identity kernel (there's really no need to scale this) """
        id = np.zeros(self.dims)
        id[self.dims[0] // 2, self.dims[1] // 2] = 1.0
        return id

    @scale
    def moffat(self, alpha: float =3.0, beta: float =2.5) -> np.ndarray:
        """ The Gaussian is a limiting case of this kernel as \beta -> \infty """
        kernel = np.zeros(self.dims)
        unit_square = (-0.5, 0.5, lambda y: -0.5, lambda y: 0.5)

        x_shift = 1.0 if self.dims[0] % 2 else 0.5
        y_shift = 1.0 if self.dims[1] % 2 else 0.5

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                # integrate on a unit square centered at the origin as the
                # function moves about it
                kernel[i][j] = dblquad(
                    lambda x, y: (beta - 1) / (pi * alpha ** 2) * \
                        (1 + ((x + i - self.dims[0] // 2 + x_shift) ** 2
                            + (y + j - self.dims[1] // 2 + y_shift) ** 2)
                         / (alpha ** 2)) ** (-beta),
                    *unit_square
                )[0]

        return kernel

    ## Add more kernels here


class GetAxis:
    """ Context manager to clear details getting specific image axes """
    def __init__(self, image_name: str, axis: Union[int, None] =0) -> None:
        self.image_name = image_name
        self.axis = axis

    def __enter__(self) -> np.ndarray:
        self._image = Image.open(self.image_name)

        # See the following for more modes
        # https://github.com/python-pillow/Pillow/blob/master/PIL/ImageMode.py#L33
        # https://github.com/python-pillow/Pillow/blob/master/PIL/Image.py#L215
        if ''.join(self._image.getbands()) == 'RGB':
            return np.flipud(np.rot90(np.array(self._image).T[self.axis], 1))
        else:
            return np.array(self._image).T[self.axis]

    def close(self) -> None:
        self._image.close()

    def __exit__(self, *args) -> None:
        self.close()


class OaAConvolve:
    """ OaA convolution algorithm """
    _convolved = None # :t Union[np.ndarray, None]

    def __init__(self, kernel: np.ndarray) -> None:
        self._KX, self._KY, *_ = kernel.shape
        self._padX, self._padY = self._KX // 2, self._KY // 2

        # wrestle this kernel into the proper representation
        self.kernel = FFT(self.invert_kernel(np.pad(
            kernel,
            [[self._padX]*2, [self._padY]*2],
            mode='constant'
        )))

    def __call__(self, another_im: Union[np.ndarray, str]) -> np.ndarray:
        """ Make instance callable on another image with the same kernel """
        if isinstance(another_im, np.ndarray):
            return self.convolve(another_im)

        elif isinstance(another_im, str):
            # composition w/CM on image (path + name); assumes axis 0
            with GetAxis(another_im) as im:
                return self.convolve(im)

        else:
            # Just a head's up
            raise ValueError(self.__class__.__name__
                + ' expected type string or np.ndarray, received %s'
                % (type(another_im))
            )

    @staticmethod
    def invert_kernel(kernel: np.ndarray) -> np.ndarray:
        """ Invert the input kernels for convolution """
        X, Y, *_ = kernel.shape
        new_kernel = np.zeros(kernel.shape)

        for i in range(X):
            for j in range(Y):
                # or vice-versa, yields the same result
                new_kernel[i, j] = kernel[(i + X // 2) % X, (j + Y // 2) % Y]

        return new_kernel

    def convolve(self, im: np.ndarray) -> np.ndarray:
        _X, _Y = im.shape
        print(_X, _Y)

        divX, divY = _X // self._KX, _Y // self._KY
        diffX = (self._KX - _X + self._KX * divX) % self._KX
        diffY = (self._KY - _Y + self._KY * divY) % self._KY

        # padding on each side, i.e. left, right, top and bottom; 
        # centered as well as possible
        right = diffX // 2
        left = diffX - right
        bottom = diffY // 2
        top = diffY - bottom

        # Pad both `im` (original image) and `self._convolved` (stores results)
        im = np.lib.pad(
            im,
            [[left, right], [top, bottom]],
            mode='constant'
        )

        self._convolved = np.zeros((left + self._padX + right + self._padX + _X,
                                    bottom + self._padY + top + self._padY + _Y))
        #self._convolved = np.zeros(im.shape)

        # Generator that parametrizes (partitions) the image into blocks
        subsets = (
            [[i * self._KX, (i + 1) * self._KX], [j * self._KY, (j + 1) * self._KY]]
                for i in range(divX) for j in range(divY)
        )

        # transform each partition and OA on conv_image
        for t, s in subsets:
            # slice and pad each array subset
            subset = np.lib.pad(
                im[slice(*t), slice(*s)],
                [[self._padX]*2, [self._padY]*2],
                mode='constant'
            )

            transf_subset = FFT(subset)

            # hadamard product and iFFT
            space = iFFT(transf_subset * self.kernel).real

            # overlap and add
            t[1] += 2 * self._padX
            s[1] += 2 * self._padY
            self._convolved[slice(*t), slice(*s)] += space

        # slice image edges off and get it back, convolved
        self._convolved = self._convolved[
            self._padX + left:self._padX + left + _X,
            self._padY + bottom:self._padY + bottom + _Y
        ]

        print(*self._convolved.shape)
        return self._convolved


def vary_scale_test() -> None:
    from scipy.misc import imsave
    im = list(filter(lambda FILENAME: FILENAME.endswith('.jpg'), iglob(DIR)))[0]
    print(im)
    k = Kernel((10, 10))

    for factor in np.linspace(0.5, 3.5, 31):
        print(factor)
        scaled = k.gaussian(sigma=3.0)
        scaled /= (factor * np.sum(scaled))

        plt.imshow(scaled, interpolation='none', cmap='gray')
        plt.show()

        convolver = OaAConvolve(scaled)
        image_name = 'spider_scaled_%.1f.png' % (factor)
        imsave(DIR[:-1] + image_name, convolver(im))


if __name__ == '__main__':
    #vary_scale_test()


    kern = Kernel((20, 20))
    convolver = OaAConvolve(kern.moffat())   # this guy is callable

    image_name = list(filter(lambda FILENAME: FILENAME.endswith('.jpg'), iglob(DIR)))[0]
    print(image_name)

    with GetAxis(image_name, axis=0) as image:
        #print(np.squeeze(image).shape)
        plt.imshow(convolver(image), interpolation='none', cmap='gray')
        #plt.imshow(kern.moffat(), interpolation='none', cmap='gray')
        plt.colorbar()
        plt.show()