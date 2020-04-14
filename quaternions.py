import numpy as np
import cv2
from scipy.fftpack import fftn, ifftn
import quaternion



def rgb_to_quaternions(image):
	q = np.zeros((len(image), len(image[0])), dtype=quaternion.quaternion)
	for i in range(len(image)):
		for j in range(len(image[0])):
			q[i][j] = np.quaternion(0,image[i][j][0],image[i][j][1],image[i][j][2])
	return q


def quaternions_to_rgb(q):
	f = np.zeros((len(q), len(q[0]), 3))
	for i in range(len(q)):
		for j in range(len(q[0])):
			t, f[i][j][0], f[i][j][1], f[i][j][2] = q[i][j].components
	return f


def quaternions_to_complex(q):
	i1 = np.zeros((len(q), len(q[0])), dtype=np.cfloat)
	i2 = np.zeros((len(q), len(q[0])), dtype=np.cfloat)
	for i in range(len(q)):
		for j in range(len(q[0])):
			i1[i][j] = q[i][j].a # ou será o contrário?
			i2[i][j] = q[i][j].b
	return i1, i2


def complex_to_quaternions(i1, i2):
	q = np.zeros((len(i1), len(i1[0])), dtype=quaternion.quaternion)
	for i in range(len(i1)):
		for j in range(len(i1[0])):
			q[i][j] = np.quaternion(i1[i][j].real,i2[i][j].imag,i2[i][j].real,i1[i][j].imag) # será?
	return q


def centralize_transform(f):
	for i in range(len(f)):
		for j in range(len(f[0])):
			f[i, j] = f[i, j] * (-1)**(i+j)
	return f


def qft(q):
	i1, i2 = quaternions_to_complex(q)
	I1 = fftn(i1)
	I2 = fftn(i2)
	Q = complex_to_quaternions(I1, I2)
	return Q


def iqft(Q):
	I1, I2 = quaternions_to_complex(Q)
	i1 = ifftn(I1)
	i2 = ifftn(I2)
	q = complex_to_quaternions(i1, i2)
	return q


def image_filtration(Q, H):
	return centralize_transform(iqft(H*Q))	


def fill_zeros(f):
	p = 2*len(f)
	q = 2*len(f[0])
	fp = np.zeros((p, q, len(f[0][0])))
	fp[:len(f), :len(f[0]), :] = f
	return fp


def ideal_filter(F, D, band):
	H = np.ones((F.shape))
	r = len(F)/2
	s = len(F[0])/2
	a, b = (1, 0) if band == 'lowPass' else (0, 1)
	for i in range(len(F)):
		for j in range(len(F[0])):
			H[i, j] = a if ((i - r)**2 + (j - s)**2)**(1/2) <= D else b
	return H


def butterworthFilter(F, D, order, band):
	H = np.ones((F.shape))
	r = len(F)/2
	s = len(F[0])/2
	for i in range(len(F)):
		for j in range(len(F[0])):
			H[i, j] = 1/(1 + ((((i - r)**2 + (j - s)**2)**(1/2))/D)**(2*order))
	return H if band == 'lowPass' else 1 - H


f = cv2.imread('images/cat.png')
fp = fill_zeros(f)
q = rgb_to_quaternions(fp)
Q = qft(centralize_transform(q))

H = ideal_filter(Q, 50, 'highPass')
g_quat = image_filtration(Q, H)[:len(f), :len(f[0])]
g_rgb = quaternions_to_rgb(g_quat)
cv2.imwrite('images/idealHighPassQuat.png', g_rgb)

H = ideal_filter(Q, 50, 'lowPass')
g_quat = image_filtration(Q, H)[:len(f), :len(f[0])]
g_rgb = quaternions_to_rgb(g_quat)
cv2.imwrite('images/idealLowPassQuat.png', g_rgb)

H = butterworthFilter(Q, 50, 2, 'highPass')
g_quat = image_filtration(Q, H)[:len(f), :len(f[0])]
g_rgb = quaternions_to_rgb(g_quat)
cv2.imwrite('images/butterworthHighPassQuat.png', g_rgb)

H = butterworthFilter(Q, 50, 2, 'lowPass')
g_quat = image_filtration(Q, H)[:len(f), :len(f[0])]
g_rgb = quaternions_to_rgb(g_quat)
cv2.imwrite('images/butterworthLowPassQuat.png', g_rgb)