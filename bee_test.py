import numpy as np
from bees import bee_search


def square_2d(input_vect):
    x1 = input_vect[0]
    x2 = input_vect[1]
    return x1 * x1 + x2 * x2


def peaks(input_vect):
    x1 = input_vect[0]
    x2 = input_vect[1]

    z1 = np.exp(-(x1 ** 2 + x2 ** 2) / 2) / (2 * np.pi)
    z2 = (np.exp(-(((x1 - 1) / 1.5) ** 2 + ((x2 - 1) / 0.5) ** 2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    z = z2 - z1

    return z


start = -100
end = 100
peaks_search_space = np.array([[start, start], [end, end]])
for i, solution in bee_search(peaks, peaks_search_space, minimize=True, max_iter=10000, step=1):
    # print(solution.__dict__)
    pass
