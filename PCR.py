import numpy as np
import matplotlib.pyplot as plt
import urllib


link = "http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat"
f=urllib.urlopen(link)
datos=f.read()

k= open("datos1.dat", "w+")
for i in range(len(datos)):
        k.write(datos[i])
	
k.close()
file_2 = np.genfromtxt("datos1.dat", delimiter=",", dtype="string")

colum_1=file_2[:,1]

txt= np.genfromtxt('datos1.dat', delimiter=',', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

colum=[]

for i in range (len(txt[0])):
        for j in range (len(txt[0])):
                lista1=txt[:,i]-np.average(txt[:,i])
                lista2=txt[:,j]-np.average(txt[:,j])
                sum=(lista1*lista2)/(len(txt)-1)
                suma=np.sum(sum)
                colum.append(suma)

matriz2=[]

j=0
while j<900:
       	k=0
	a=[]
       	while k<30:
       		a.append(colum[j])
       		k=k+1
		j=j+1
	matriz2.append(a)
	a=[]
print "LA matriz de covariaza es:"
print matriz2
#sacar autovectores y autovalores

x,y= np.linalg.eig(matriz2)
print "MEAN"
print "radio"
print "Autovector",y[0]," Autovalores", x[0]
print "Textura" 
print "Autovector",y[1]," Autovalores", x[1]
print "Perimetro"
print"Autovector",y[2]," Autovalores", x[2]
print "Area"
print"Autovector",y[3]," Autovalores", x[3]
print "Smoothness"
print"Autovector",y[4]," Autovalores", x[4]
print "Compactness"
print"Autovector",y[5]," Autovalores", x[5]
print "Cancavidad"
print"Autovector",y[6]," Autovalores", x[6]
print "Puntos concavos"
print"Autovector",y[7]," Autovalores", x[7]
print "Simetria"
print"Autovector",y[8]," Autovalores", x[8]
print "Fractal dimension"
print"Autovector",y[9]," Autovalor", x[9]

print" "
print"Los parametros mas mportantes son el radio (",x[0], ")  y la textura (", x[1], ") debido a que tienen el mayor valor."



print txt
eje_x=np.dot(txt,y[0])
eje_y=np.dot(txt, y[1])

print y[0],y[1]
#plt.scatter(eje_x, eje_y)
#plt.show()
