import numpy as np
from scipy.fftpack import fftn, ifftn
import quaternion
import time


def rgb_to_quaternions(image):
    def aux(axis):
      return np.quaternion(0, axis[0], axis[1], axis[2])
    q = np.apply_along_axis(aux, 2, image)
    return q


def quaternions_to_rgb(q):
    return quaternion.as_float_array(q)[:,:,1:]


def quaternions_to_complex(q):
    float_array = quaternion.as_float_array(q)
    i1 = float_array[..., 0] + 1j * float_array[..., 2]
    i2 = float_array[..., 1] + 1j * float_array[..., 3]
    return i1, i2


def complex_to_quaternions(i1, i2):
    dim_array = list(i1.shape)
    dim_array.append(4)

    float_array = np.zeros(tuple(dim_array))

    float_array[..., 0] = np.real(i1)
    float_array[..., 1] = np.real(i2)
    float_array[..., 2] = np.imag(i1)
    float_array[..., 3] = np.imag(i2)

    return quaternion.as_quat_array(float_array)


def centralize_transform(f):
    mask = [1, -1]*(len(f[0])//2)
    mask += mask if len(f[0]) % 2 != 0 else [-1*i for i in mask]
    mask = np.array(mask*(len(f)//2)).reshape(f.shape)
    return f*mask


def qfft(q):
    i1, i2 = quaternions_to_complex(q)
    I1 = fftn(i1)
    I2 = fftn(i2)
    Q = complex_to_quaternions(I1, I2)
    return Q


def iqfft(Q):
    I1, I2 = quaternions_to_complex(Q)
    i1 = ifftn(I1)
    i2 = ifftn(I2)
    q = complex_to_quaternions(i1, i2)
    return q


def frequency_filtration(Q, H):
    start = time.time()
    filtered_frequency = iqfft(H * Q)
    end = time.time()
    return centralize_transform(filtered_frequency), end - start


def fill_zeros(f):
    p = 2 * len(f)
    q = 2 * len(f[0])
    fp = np.zeros((p, q, len(f[0][0])))
    fp[:len(f), :len(f[0]), :] = f
    return fp


def ideal_filter(F, band, D=50):
    H = np.ones(F.shape)
    r = len(F) / 2
    s = len(F[0]) / 2
    a, b = (1, 0) if band == 'low_pass' else (0, 1)
    for i in range(len(F)):
        for j in range(len(F[0])):
            H[i, j] = a if ((i - r) ** 2 + (j - s) ** 2) ** (1 / 2) <= D else b
    return H


def butterworth_filter(F, band, D=50, order=2):
    H = np.ones(F.shape)
    r = len(F) / 2
    s = len(F[0]) / 2
    for i in range(len(F)):
        for j in range(len(F[0])):
            H[i, j] = 1 / (1 + ((((i - r) ** 2 + (j - s) ** 2) ** (1 / 2)) / D) ** (2 * order))
    return H if band == 'low_pass' else 1 - H


def quaternion_filtration(image, filter):
    f = fill_zeros(image)
    q = rgb_to_quaternions(f)
    Q = qfft(centralize_transform(q))
    if filter['name'] == 'ideal_filter':
        H = ideal_filter(Q, filter['band'])
    elif filter['name'] == 'butterworth_filter':
        H = butterworth_filter(Q, filter['band'])
    g_quat, t = frequency_filtration(Q, H)
    g_quat = g_quat[:len(image), :len(image[0])]
    g_rgb = quaternions_to_rgb(g_quat)
    return g_rgb.astype("int"), t
