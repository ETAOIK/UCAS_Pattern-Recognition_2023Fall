# Pattern Recognition 2023 Fall
# Chenkai GUO 
# Date: 2023.10.30

import numpy as np
import matplotlib.pyplot as plt

samples = [4.6019, 5.2564, 5.2200, 3.2886, 3.7942,
          3.2271, 4.9275, 3.2789, 5.7019, 3.9945,
          3.8936, 6.7906, 7.1624, 4.1807, 4.9630,
          6.9630, 4.4597, 6.7175, 5.8198, 5.0555,
          4.6469, 6.6931, 5.7111, 4.3672, 5.3927,
          4.1220, 5.1489, 6.5319, 5.5318, 4.2403,
          5.3480, 4.3022, 7.0193, 3.2063, 4.3405,
          5.7715, 4.1797, 5.0179, 5.6545, 6.2577,
          4.0729, 4.8301, 4.5283, 4.8858, 5.3695,
          4.3814, 5.8001, 5.4267, 4.5277, 5.2760]


def Parzen_1d(samples: list,
              kernel: chr,
              h: int,
              resolution = 0.1,
              ):
    """set initial parameters"""
    X = [x for x in np.arange(min(samples)-1, max(samples)+1, resolution)]
    Y = [zero for zero in np.zeros(len(X))]

    if kernel == 'square':
        for element in samples:
            for point in range(len(X)):
                if abs(element - X[point]) < (h / 2):
                    Y[point] += 1 / (h * len(samples))
    elif kernel == 'gaussian':
        hn = h / (len(samples) ** 0.5)
        for element in samples:
            for point in range(len(X)):
                a = (element - X[point]) / hn
                Y[point] += np.exp(-0.5 * a * a) / ((2 * np.pi) ** 0.5) / (len(samples) * hn)

    return Y

if __name__ == '__main__' :
    X = [x for x in np.arange(min(samples) - 1, max(samples) + 1, 0.1)]

    y_square1 = Parzen_1d(samples, 'square', 0.5)
    y_square2 = Parzen_1d(samples, 'square', 1.0)
    y_square3 = Parzen_1d(samples, 'square', 3.0)

    y_gaussian1 = Parzen_1d(samples, 'gaussian', 0.5)
    y_gaussian2 = Parzen_1d(samples, 'gaussian', 1.0)
    y_gaussian3 = Parzen_1d(samples, 'gaussian', 3.0)

    plt.figure(figsize=(11, 5))

    plt.subplot(121)
    plt.plot(X, y_square1, label="h=0.5")
    plt.plot(X, y_square2, label="h=1.0")
    plt.plot(X, y_square3, label="h=3.0")
    plt.title('Parzen Window(square)')
    plt.legend()

    plt.subplot(122)
    plt.plot(X, y_gaussian1, label="h=0.5")
    plt.plot(X, y_gaussian2, label="h=1.0")
    plt.plot(X, y_gaussian3, label="h=3.0")
    plt.title('Parzen Window(gaussian)')
    plt.legend()
    
    plt.savefig('parzen_results.png')
    plt.show()

