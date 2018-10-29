import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy import interpolate


dat= np.genfromtxt("signal.dat", usecols=(0,2))

n = len(dat) # numbero de puntos en el intervalo
t = dat[:,0]
y = dat[:,1]
dt=t[1]-t[0]
plt.figure()
plt.plot(t,y)
#plt.show()


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

dat_in= np.genfromtxt("incompletos.dat", usecols=(0,2))

n_2 = len(dat_in) # numbero de puntos en el intervalo                                                     
t_2= dat_in[:,0]
y_2= dat_in[:,1]
dt_2=t_2[1]-t_2[0]
plt.figure()
plt.plot(t_2,y_2)
plt.show()                                                                                          


G_N2=[]
i_s=[]
for i in range (n_2):
    gn=0.0
    i_s.append(i)
    for j in range (n):
        h= y[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N2.append(gn)

print G_N2

h_2=fft.fftfreq(n_2,dt_2)
transformada_2=abs(np.real(G_N2))
print np.size(y_2)
print np.size(t_2)
print np.size(h_2)
print np.size(transformada_2)

plt.plot(h_2, transformada_2)
plt.show()



x_interpolacion= np.linspace(dat_in[0,0], dat_in[(n_2-1),0],512)
h_i= (dat_in[0,0] - dat_in[(n_2-1),0])/512
fcuadratica=[]
fcubica=[]
#funcion que hace la interpolacion y grafica la lineal, cuadratica y cubica
def interpolacion (dat_in, x_interpolacion,fcuadratica,fcubica): 
#tiene com oparametro el archivo y el arrray como pide el enunciado y ademaslas 3 listas con los puntos interpolados ya que los voya necesitar para el punto b
    x=dat_in[:,0]
    y=dat_in[:,1]
    cuadratica=interpolate.interp1d(x,y, kind="quadratic")
    cubica=interpolate.interp1d(x,y, kind="cubic")
    f_cuadratica=cuadratica(x_interpolacion)
    f_cubica=cubica(x_interpolacion)
    #convierto las interpolaciones en listas y luego agregos los valores a las listas de afuera para usarlas en punto b
    f_cuadratica=f_cuadratica.tolist()
    f_cubica=f_cubica.tolist()
    for i in range (len(f_cuadratica)):
        fcuadratica.append(f_cuadratica[i])
        fcubica.append(f_cubica[i])
    

interp= interpolacion(dat_in, x_interpolacion,fcuadratica,fcubica)


fcuadratica=np.array(fcuadratica)
fcubica=np.array(fcubica)
print np.size(fcuadratica)

n_3 = 512 # numbero de puntos en el intervalo                                                              
t = dat[:,0]
y_3= fcuadratica
G_N3=[]
i_s=[]
for i in range (n_3):
    gn=0.0
    i_s.append(i)
    for j in range (n):
        h= y[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N3.append(gn)

print G_N3

h_3=fft.fftfreq(n_3,h_i)
transformada_3=abs(np.real(G_N3))


n_4 = 512 # numbero de puntos en el intervalo                                                                 
t = dat[:,0]
y_4= fcubica
G_N4=[]
i_s=[]
for i in range (n_4):
    gn=0.0
    i_s.append(i)
    for j in range (n):
        h= y[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N4.append(gn)

h= fft.fftfreq(n,dt)
h_4=fft.fftfreq(n_4,h_i)
transformada_4=abs(np.real(G_N4))
plt.figure()
plt.plot(h_3,transformada_3)
plt.plot(h_4,transformada_4)
plt.plot(h,transformada)
plt.show()
