import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy import interpolate

#tarer datos del signal.dat
dat= np.genfromtxt("signal.dat", usecols=(0,2))

n = len(dat) # numbero de puntos en el intervalo
t = dat[:,0]
y = dat[:,1]
dt=t[1]-t[0]

#grafia de datos
plt.figure()
plt.plot(t,y, c="lime")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Datos T vs A signal.dat")
plt.savefig("QuinteroLiliana_signal.pdf")

#implementacion propia Fourier:
G_N=[]
i_s=[]
for i in range (n):
    gn=0.0
    i_s.append(i)
    for j in range (n):
        h= y[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N.append(gn)

h=fft.fftfreq(n,dt) #sacar la frecuencia
transformada=abs(np.real(G_N))

#plot transformada
plt.figure()
plt.plot(h,transformada, c="darkmagenta")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Transformada de Fourier Signal.dat F vs A")
plt.savefig("QuinteroLiliana_TF.pdf")

print "Las frecuencias principales de la funcion son :384.261 Hz, 362.544 Hz, 139.695 Hz y 0"

#el filtro para la transformada pasa bajos
def filtro(trans, frec):
    for i in range(len(frec)):
        if(abs(frec[i])>1000):
            trans[i]=0
    return trans

f=filtro(G_N,h)

#transformada inversa de la filtrada
onda_filtrada= np.fft.ifft(f)
filtrada=(np.real(onda_filtrada))


plt.figure()
plt.plot(t,filtrada, c="gold")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud Filtrada")
plt.title("Senial filtrada")
plt.savefig("QuinteroLiliana_filtrada.pdf")

#datos incompletos
dat_in= np.genfromtxt("incompletos.dat", usecols=(0,2))

n_2 = len(dat_in) # numbero de puntos en el intervalo                                                     
t_2= dat_in[:,0]
y_2= dat_in[:,1]
dt_2=t_2[1]-t_2[0]                                                                                         

#transformada datos incompletos
G_N2=[]
i_s=[]
for i in range (n_2):
    gn=0.0
    i_s.append(i)
    for j in range (n):
        h= y[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N2.append(gn)

h_2=fft.fftfreq(n_2,dt_2)
transformada_2=abs(np.real(G_N2))


print "Los datos incompletos no se pueden usar ya que para poder usar la transformada de Fourier se asume que los puntos estan todos distanciados igualmente y como los datos se encuentran incompletos la distancia entre estos varia entre cada punto y no podemos hallar la transformada."

#interpolacion
x_interpolacion= np.linspace(dat_in[0,0], dat_in[(n_2-1),0],512)
h_i= (dat_in[0,0] - dat_in[(n_2-1),0])/512
fcuadratica=[]
fcubica=[]
#funcion que hace la interpolacion cuadratica y cubica
def interpolacion (dat_in, x_interpolacion,fcuadratica,fcubica): 
    x=dat_in[:,0]
    y=dat_in[:,1]
    cuadratica=interpolate.interp1d(x,y, kind="quadratic")
    cubica=interpolate.interp1d(x,y, kind="cubic")
    f_cuadratica=cuadratica(x_interpolacion)
    f_cubica=cubica(x_interpolacion)
    f_cuadratica=f_cuadratica.tolist()
    f_cubica=f_cubica.tolist()
    for i in range (len(f_cuadratica)):
        fcuadratica.append(f_cuadratica[i])
        fcubica.append(f_cubica[i])
    

interp= interpolacion(dat_in, x_interpolacion,fcuadratica,fcubica)


fcuadratica=np.array(fcuadratica)
fcubica=np.array(fcubica)

#transformadas de Fourier para las interpolaciones

n_3 = 512 # numbero de puntos en el intervalo                                                              
t = dat[:,0]
y_3= fcuadratica
G_N3=[]
for i in range (n_3):
    gn=0.0
    for j in range (n):
        h= y_3[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N3.append(gn)


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
        h= y_4[j]*np.exp(-1j*2*np.pi*j*(float(i)/n))
        gn=gn+h
    G_N4.append(gn)

h= fft.fftfreq(n,dt)
h_4=fft.fftfreq(n_4,h_i)
transformada_4=abs(np.real(G_N4))

#plotear con subplots las tres transformadas de fourier

plt.figure()
plt.subplots_adjust(hspace=0.5)
plt.subplot(311)
plt.plot(h_3,transformada_3, label="Interp. Cuadratica", c="crimson")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.legend(loc="best")
plt.title("Transformadas de Fourier de Interpolaciones y transformada normal")
plt.subplot(312)
plt.plot(h_4,transformada_4, label="Interp. Cubica", c="aqua")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.legend(loc="best")
plt.subplot(313)
plt.plot(h,transformada, label= "Transformada", c="indigo")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.legend(loc="best")
plt.savefig("QuinteroLiliana_TF_interpola.pdf")

print "En la transformada de las interpolaciones se elimina uno de los picos principales sim embargo alrededor de los picos mas altos hay mas maplitudes pequenias que en la transformada normal no se encuentran, se concervan las amplitudes de lso picos principales sin embargo las interpolaciones 'matan' tambien los picos pequenios para las frecuencias mayores como una especia de mini-filtro dodne se disminuyen las amplitudes de estas frecuencias."


#filtro para <500
def filtro2(trans, frec):
    for i in range(len(frec)):
        if(abs(frec[i])<500):
            trans[i]=0
    return trans

#filtro 1000
n_filt= filtro(G_N, h) 
cua_filt= filtro(G_N3,h)
cub_filt= filtro(G_N4, h)
f1=np.fft.ifft(n_filt)
f2=np.fft.ifft(cua_filt)
f3=np.fft.ifft(cub_filt)

#filtro 500
n_filt_2= filtro(G_N, h)
cua_filt_2= filtro(G_N3,h_3)
cub_filt_2= filtro(G_N4, h_4)
f1_500= fft.ifft(n_filt_2)
f2_500= fft.ifft(cua_filt_2)
f3_500= fft.ifft(cub_filt_2)

#plotear para ambos filtros los tres metodos
plt.figure()
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(t, f1, c='dodgerblue', label="Transformada")
plt.plot(t, f2, c='chartreuse', label="Cuadratica")
plt.plot(t, f3, 'r--', label="Cubica")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Senial filtrada con filtro <500 y >1000 \n Filtro >1000")
plt.legend(loc="best")
plt.subplot(212)
plt.plot(t, f1_500, c='dodgerblue', label="Transformada")
plt.plot(t, f2_500, c='chartreuse', label="Cuadratica")
plt.plot(t, f3_500, 'r--', label="Cubica")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Filtro <500")
plt.legend(loc="best")
plt.savefig("QuinteroLiliana_2Filtros.pdf")



