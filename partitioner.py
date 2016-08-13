#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def divider(arr_dims, coreNum=1):
    """ Get a bunch of iterable ranges; 
    Example input: [[[0, 24], [15, 25]]]"""
    if (coreNum == 1):
        return arr_dims

    elif (coreNum < 1):
        raise ValueError(\
      'partitioner expected a positive number of cores, got %d'\
                    % coreNum
        )

    elif (coreNum % 2):
        raise ValueError(\
      'partitioner expected an even number of cores, got %d'\
                    % coreNum
        )
    
    total = []

    # Split each coordinate in arr_dims in _half_
    for arr_dim in arr_dims:
        dY = arr_dim[0][1] - arr_dim[0][0]
        dX = arr_dim[1][1] - arr_dim[1][0]
        
        if ((coreNum,)*2 > (dY, dX)):
            coreNum = max(dY, dX)
            coreNum -= 1 if (coreNum % 2 and coreNum > 1) else 0

        new_c1, new_c2, = [], []

        if (dY >= dX):
            # Subimage height is greater than its width
            half = dY // 2
            new_c1.append([arr_dim[0][0], arr_dim[0][0] + half])
            new_c1.append(arr_dim[1])
            
            new_c2.append([arr_dim[0][0] + half, arr_dim[0][1]])
            new_c2.append(arr_dim[1])

        else:
            # Subimage width is greater than its height
            half = dX // 2
            new_c1.append(arr_dim[0])
            new_c1.append([arr_dim[1][0], half])

            new_c2.append(arr_dim[0])
            new_c2.append([arr_dim[1][0] + half, arr_dim[1][1]])

        total.append(new_c1), total.append(new_c2)

    # If the number of cores is 1, we get back the total; Else,
    # we split each in total, etc.; it's turtles all the way down
    return divider(total, coreNum // 2)
        

if __name__ == '__main__':
    import numpy as np
    X = np.random.randn(25 - 1, 36 - 1)
    dims = [zip(*([0, 0], list(X.shape)))]
    dims = [list(j) for i in dims for j in dims[0] if type(j) != list]
    print divider([dims], 2)