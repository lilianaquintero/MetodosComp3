import numpy as np 
import matplotlib.pyplot as plt 

file_2 = np.genfromtxt("datos.dat",delimiter=",", dtype="string")

colum_1=file_2[:,1]

txt= np.genfromtxt('datos.dat', delimiter=',', usecols=(2,3,4,5,6,7,8,9,10,11))


colum=[]

for i in range (len(txt[0])):
        for j in range (len(txt[0])):
                lista1=txt[:,i]-np.average(txt[:,i])
                lista2=txt[:,j]-np.average(txt[:,j])
                sum=(lista1*lista2)/(len(txt)-1)
                suma=np.sum(sum)
                colum.append(suma)


print np.size(colum)


