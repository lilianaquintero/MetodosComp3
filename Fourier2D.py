import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from matplotlib.colors import LogNorm

img= plt.imread("arbol.png")

transformada= np.fft.fft2(img) #transformada de Fourier
shifteada= np.fft.fftshift(transformada) 
frec=np.fft.fftfreq(len(transformada[0]))

#plot transformada en escala logaritmicaya que uestra bien el centro y a escala
plt.figure()
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(frec, transformada)
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Transformada de Fourier de la imagen \n Transf. Frec vs Amplitud")
plt.subplot(212)
plt.imshow(10*np.log10(abs(shifteada)),cmap="gray")
plt.colorbar()
plt.savefig("QuinteroLiliana_FT2D.pdf")

t_filtrar=np.copy(transformada)
#filtro para eliminar el ruido el cual a partir del plot de Ampl vs Frec s puede ver que corresponde a valores de amplitud entre 4100 y 4120

for i in range(len(frec)):
	for j in range(len(frec)):
        	if((abs(t_filtrar[i,j])>4100) and (abs(t_filtrar[i,j])<4120)):
            		t_filtrar[i,j]=0


shift_f=np.fft.fftshift(t_filtrar) 

plt.figure()
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(frec, t_filtrar)
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Filtro de la transformada \n Filtro. Frec vs Amplitud")
plt.subplot(212)
plt.imshow((abs(shift_f)), cmap="gray", norm=LogNorm())
plt.colorbar()
plt.savefig("QuinteroLiliana_FT2D_filtrada.pdf")


filt_im= np.fft.ifft2(t_filtrar)
filt= np.array(filt_im)
plt.imsave("QuinteroLiliana_Imagen_filtrada.pdf", np.real(filt), cmap="gray")






