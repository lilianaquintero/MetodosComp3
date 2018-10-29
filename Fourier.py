import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

dat= np.genfromtxt("signal.dat", usecols=(0,2))
print dat

n = len(dat) # numbero de puntos en el intervalo
t = dat[:,0]
y = dat[:,1]

plt.figure()
plt.plot(t,y)
plt.show()


G_N=[]
i_s=[]
for i in range (n):
    gn=0.0
    i_s.append(i)
    for j in range (n):
        h= y[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N.append(gn)

print G_N

h=fft.fftfreq(len(G_N))

plt.figure()
plt.plot(h,np.real(G_N))
plt.show()


