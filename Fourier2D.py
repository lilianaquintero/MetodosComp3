import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
#from scipy import ndimage, misc 


img= plt.imread("arbol.png")
print np.shape(img)

transformada= np.fft.fft2(img)
trans_shifteada= np.fft.fftshift(img) 
frec=np.fft.fftfreq(len(transformada[0]))

plt.plot(frec, transformada)
plt.show()

def filtro(trans, frec):
    for i in range(len(frec)):
	for j in range(len(frec)):
        	if((abs(trans[i,j])>1000) and (abs(trans[i,j])<5000)):
            		trans[i,j]=0
    return trans

f=filtro(transformada,frec)

#plt.plot(frec,f)
plt.imshow(10*np.log10(abs(f)))
plt.colorbar()
plt.show()

filt_im= np.fft.ifft(f)
filt= np.array(filt_im)

plt.imsave("bla.pdf", filt.real)






