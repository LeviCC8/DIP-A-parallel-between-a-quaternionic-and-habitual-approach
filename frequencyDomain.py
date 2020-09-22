import numpy as np
from scipy.fftpack import fftn, ifftn
import time


def fill_zeros(f):
    p = 2 * len(f)
    q = 2 * len(f[0])
    fp = np.zeros((p, q, len(f[0][0])))
    fp[:len(f), :len(f[0]), :] = f
    return fp


def centralize_transform(f):
    aux = [1, -1] * (len(f[0]) // 2)
    aux += aux if len(f[0]) % 2 != 0 else [-1 * i for i in aux]
    aux = np.array(aux * (len(f) // 2)).reshape(f.shape[:-1])
    mask = np.zeros(f.shape)
    mask[:, :, 0] = aux
    mask[:, :, 1] = aux*(-1)
    mask[:, :, 2] = aux
    return f * mask


def frequency_filtration(F, H):
    start = time.time()
    filtered_frequency = ifftn(H * F)
    end = time.time()
    return centralize_transform(filtered_frequency.real), end - start


def ideal_filter(F, band, D=50):
    H = np.ones(F.shape)
    r = len(F) / 2
    s = len(F[0]) / 2
    a, b = (1, 0) if band == 'low_pass' else (0, 1)
    for i in range(len(F)):
        for j in range(len(F[0])):
            H[i, j, 0] = a if ((i - r) ** 2 + (j - s) ** 2) ** (1 / 2) <= D else b
    H[:, :, 1] = H[:, :, 2] = H[:, :, 0]
    return H


def butterworth_filter(F, band, D=50, order=2):
    H = np.ones(F.shape)
    r = len(F) / 2
    s = len(F[0]) / 2
    for i in range(len(F)):
        for j in range(len(F[0])):
            H[i, j, 0] = 1 / (1 + ((((i - r) ** 2 + (j - s) ** 2) ** (1 / 2)) / D) ** (2 * order))
    H[:, :, 1] = H[:, :, 2] = H[:, :, 0]
    return H if band == 'low_pass' else 1 - H


def habitual_filtration(image, filter):
    f = fill_zeros(image)
    F = fftn(centralize_transform(f))
    if filter['name'] == 'ideal_filter':
        H = ideal_filter(F, filter['band'])
    elif filter['name'] == 'butterworth_filter':
        H = butterworth_filter(F, filter['band'])
    g, t = frequency_filtration(F, H)
    g = g[:len(image), :len(image[0]), :]
    return g.astype("int"), t
