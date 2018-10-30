import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft


img= plt.imread("arbol.png")
print np.shape(img)

transformada= np.fft.fft2(img)
trans_shifteada= np.fft.fftshift(img) 
frec=np.fft.fftfreq(len(transformada[0]))

plt.figure()
plt.subplot(211)
plt.plot(frec, transformada)
plt.subplot(212)
plt.imshow(abs(transformada), )
plt.colorbar()
plt.show()

def filtro(trans, frec):
    for i in range(len(frec)):
	for j in range(len(frec)):
        	if((abs(trans[i,j])>4000) and (abs(trans[i,j])<5000)):
            		trans[i,j]=0
    return trans

f=filtro(transformada,frec)

plt.figure()
plt.subplot(211)
plt.plot(frec, f)
plt.subplot(212)
plt.imshow(abs(f))
plt.colorbar()
plt.show()


filt_im= np.fft.ifft2(f)
filt= np.array(filt_im)
print np.shape(filt)
plt.imsave("bla_11.pdf", np.real(filt), cmap="gray")






