import numpy as np

#AND Gate
AND_X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
AND_Y = np.array([0,0,0,1])

#OR Gate
OR_X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
OR_Y = np.array([0,1,1,1])

#NOT Gate
NOT_X = np.array([
    [0],
    [1]
])
NOT_Y = np.array([1,0])

#XOR Gate
XOR_X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
XOR_Y = np.array([0,1,1,0])