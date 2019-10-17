import numpy as np
from scipy.fftpack import fftn, ifftn
import cv2


def fillZeros(f):
	p = 2*len(f)
	q = 2*len(f[0])
	fp = np.zeros((p, q, len(f[0][0])))
	fp[:len(f), :len(f[0]), :] = f
	return fp

def centralizeTransform(f):
	for k in range(3):
		for i in range(len(f)):
			for j in range(len(f[0])):
				f[i, j, k] = f[i, j, k] * (-1)**(i+j)
	return f

def imageFiltration(F, H):
	return centralizeTransform(ifftn(H*F).real)

def idealFilter(F, D, band):
	H = np.ones((F.shape))
	r = len(F)/2
	s = len(F[0])/2
	a, b = (1, 0) if band == 'lowPass' else (0, 1)
	for i in range(len(F)):
		for j in range(len(F[0])):
			H[i, j, 0] = a if ((i - r)**2 + (j - s)**2)**(1/2) <= D else b
	H[:,:,1] = H[:,:,2] = H[:,:,0]
	return H

def butterworthFilter(F, D, order, band):
	H = np.ones((F.shape))
	r = len(F)/2
	s = len(F[0])/2
	for i in range(len(F)):
		for j in range(len(F[0])):
			H[i, j, 0] = 1/(1 + ((((i - r)**2 + (j - s)**2)**(1/2))/D)**(2*order))
	H[:,:,1] = H[:,:,2] = H[:,:,0]
	return H if band == 'lowPass' else 1 - H

f = cv2.imread("cat.png", -1)
fp = centralizeTransform(fillZeros(f))
F = fftn(fp) 

H = idealFilter(F, 50, 'highPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('idealHighPass.png', g)

H = idealFilter(F, 50, 'lowPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('idealLowPass.png', g)

H = butterworthFilter(F, 50, 2, 'highPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('butterworthHighPass.png', g)

H = butterworthFilter(F, 50, 2, 'lowPass')
g = imageFiltration(F, H)[:len(f), :len(f[0]), :]
cv2.imwrite('butterworthLowPass.png', g)