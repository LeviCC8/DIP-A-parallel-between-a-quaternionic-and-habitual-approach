import numpy as np
import cv2


def pixelFiltration(f, w, x, y):
	a = int(len(w)/2)
	b = int(len(w[0])/2)
	g = np.array(list(map(lambda s: list(map(lambda t: w[a+s, b+t]*f[x+s, y+t], range(-b, b+1))),range(-a, a+1)))).sum()
	return g

def imageFiltration(f, w):
	g = np.zeros(f.shape)
	g[:,:,3] = f[:,:,3] # layer A
	constantX = len(w) -1
	constantY = len(w[0]) -1
	h = np.zeros((len(f) + 2*constantX ,len(f[0]) + 2*constantY, 3))
	h[constantX : constantX + len(f), constantY : constantY + len(f[0]), :] = f[:,:,:3]
	for z in range(3):
		layer = h[:,:,z]
		for x in range(len(g)): 
			for y in range(len(g[0])):
				g[x,y,z] = pixelFiltration(layer, w, x + constantX, y + constantY)
	return g


w = np.ones((3,3))*1/9
f = cv2.imread("images/cat.png", -1)
g = imageFiltration(f,w)
cv2.imwrite('images/pixelAverage.png', g)