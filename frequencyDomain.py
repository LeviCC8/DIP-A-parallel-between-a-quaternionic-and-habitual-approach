import numpy as np
from scipy.fftpack import fftn, ifftn
import cv2


def fill_zeros(f):
    p = 2 * len(f)
    q = 2 * len(f[0])
    fp = np.zeros((p, q, len(f[0][0])))
    fp[:len(f), :len(f[0]), :] = f
    return fp


def centralize_transform(f):
    for k in range(3):
        for i in range(len(f)):
            for j in range(len(f[0])):
                f[i, j, k] = f[i, j, k] * (-1) ** (i + j)
    return f


def image_filtration(F, H):
    return centralize_transform(ifftn(H * F).real)


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
    g = image_filtration(F, H)[:len(f), :len(f[0]), :]
    return g


'''
f = cv2.imread("images/cat.png")
fp = fillZeros(f)
F = fftn(centralizeTransform(fp)) 

H = idealFilter(F, 50, 'highPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('images/idealHighPass.png', g)

H = idealFilter(F, 50, 'lowPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('images/idealLowPass.png', g)

H = butterworthFilter(F, 50, 2, 'highPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('images/butterworthHighPass.png', g)

H = butterworthFilter(F, 50, 2, 'lowPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('images/butterworthLowPass.png', g)
'''
