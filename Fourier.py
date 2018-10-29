import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

dat= np.genfromtxt("signal.dat", usecols=(0,2))
print dat

n = len(dat) # numbero de puntos en el intervalo
t = dat[:,0]
y = dat[:,1]
dt=t[1]-t[0]
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

h=fft.fftfreq(n,dt)
transformada=abs(np.real(G_N))
plt.figure()
plt.plot(h,transformada)
plt.show()

def filtro(trans, frec):
    for i in range(len(frec)):
        if(abs(frec[i])>1000):
            trans[i]=0
    return trans

f=filtro(transformada,h)

plt.plot(h,f)
plt.show()

onda_filtrada= np.fft.ifft(f)
filtrada=abs(np.real(onda_filtrada))

plt.figure()
plt.plot(t,y)
plt.plot(t,filtrada)
plt.show()

