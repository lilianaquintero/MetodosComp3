import numpy as np
import matplotlib.pyplot as plt
import urllib



#obtengo los datos del link enviado y los paso a un .dat
link = "http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat"
f=urllib.urlopen(link)
datos=f.read()

k= open("WDBC.dat", "w+")
for i in range(len(datos)):
        k.write(datos[i])
	
k.close()

#abro el .dat y leo las columnas que necesito
file_2 = np.genfromtxt("WDBC.dat", delimiter=",", dtype="string")

#esta es la columna de las letras
colum_1=file_2[:,1]

txt= np.genfromtxt('WDBC.dat', delimiter=',', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

colum=[]

#metodo de covariazas
for i in range (len(txt[0])):
        for j in range (len(txt[0])):
                lista1=txt[:,i]-np.average(txt[:,i])
                lista2=txt[:,j]-np.average(txt[:,j])
                sum=(lista1*lista2)/(len(txt)-1)
                suma=np.sum(sum)
                colum.append(suma)

matriz2=[]
#transformo los datos en una matriz ya que el metodo de fourier los saca como elementos de una lista
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
print "La matriz de covariaza es:"
print matriz2
#sacar autovectores y autovalores e imprimirlos
x,y= np.linalg.eig(matriz2)

print " "

print "MEAN"
print " "
print "radio"
print "Autovector",y[0]," Autovalores", x[0]
print " "
print "Textura" 
print "Autovector",y[1]," Autovalores", x[1]
print " "
print "Perimetro"
print"Autovector",y[2]," Autovalores", x[2]
print " "
print "Area"
print"Autovector",y[3]," Autovalores", x[3]
print " "
print "Smoothness"
print"Autovector",y[4]," Autovalores", x[4]
print " "
print "Compactness"
print"Autovector",y[5]," Autovalores", x[5]
print " "
print "Cancavidad"
print"Autovector",y[6]," Autovalores", x[6]
print " "
print "Puntos concavos"
print"Autovector",y[7]," Autovalores", x[7]
print " "
print "Simetria"
print"Autovector",y[8]," Autovalores", x[8]
print " "
print "Fractal dimension"
print"Autovector",y[9]," Autovalor", x[9]
print " "
print " "
print "STANDARD ERROR"
print " "
print "radio"
print "Autovector",y[10]," Autovalores", x[10]
print " "
print "Textura" 
print "Autovector",y[11]," Autovalores", x[11]
print " "
print "Perimetro"
print"Autovector",y[12]," Autovalores", x[12]
print " "
print "Area"
print"Autovector",y[13]," Autovalores", x[13]
print " "
print "Smoothness"
print"Autovector",y[14]," Autovalores", x[14]
print " "
print "Compactness"
print"Autovector",y[15]," Autovalores", x[15]
print " "
print "Cancavidad"
print"Autovector",y[16]," Autovalores", x[16]
print " "
print "Puntos concavos"
print"Autovector",y[17]," Autovalores", x[17]
print " "
print "Simetria"
print"Autovector",y[18]," Autovalores", x[18]
print " "
print "Fractal dimension"
print"Autovector",y[19]," Autovalor", x[19]
print " "
print " "
print "WORST"
print " "
print "radio"
print "Autovector",y[20]," Autovalores", x[20]
print " "
print "Textura" 
print "Autovector",y[21]," Autovalores", x[21]
print " "
print "Perimetro"
print"Autovector",y[22]," Autovalores", x[22]
print " "
print "Area"
print"Autovector",y[23]," Autovalores", x[23]
print " "
print "Smoothness"
print"Autovector",y[24]," Autovalores", x[24]
print " "
print "Compactness"
print"Autovector",y[25]," Autovalores", x[25]
print " "
print "Cancavidad"
print"Autovector",y[26]," Autovalores", x[26]
print " "
print "Puntos concavos"
print"Autovector",y[27]," Autovalores", x[27]
print " "
print "Simetria"
print"Autovector",y[28]," Autovalores", x[28]
print " "
print "Fractal dimension"
print"Autovector",y[29]," Autovalor", x[29]
print" "

print"Los parametros mas importantes son el radio y la textura ya que los otros eigenvector son similares a e estos por tanto no son vectores independientes, estos y el perimetro son independientes y radio y textura son los que tienen los eigenvalues mas grandes."



#saco todos los datos en coordenadas de PC1 y PC2
eje_x=np.dot(txt,y[1])
eje_y=np.dot(txt, y[0])
B_x=[]
B_y=[]
M_x=[]
M_y=[]
#separacion malignos y benignos
for i in range (len(eje_x)):
	if(colum_1[i]=="M"):
		M_x.append(eje_x[i])
		M_y.append(eje_y[i])
	else:
		B_x.append(eje_x[i])
		B_y.append(eje_y[i])

plt.figure()
plt.scatter(B_x, B_y, c='green', label="Benignos", s=1)
plt.scatter(M_x, M_y, c='red', label="Malignos", s=1)
plt.xlabel("PC1: Textura")
plt.ylabel("PC2: Radio")
plt.title("Proyeccion datos en PC1 y PC2")
plt.legend(loc="best")
plt.savefig("QuinteroLiliana_PCA.pdf")

print " "
print "El metodo es util para determinar si la muestra es benigna o maligna ya que en la grafica se puede observar un 'punto de corte' de los datos donde estando en cierta zona todos los datos son benignos y en otro todos son malignos asi con certeza se puede determinar en estos casos si es benigno o maligno, sin empargo para la zona central donde se mezclan algunos datos de benigno y maligno se podria quiza observar la tercera componente independiente para ver su papel en los datos, pero fuera de esa pequenia zona donde hay ambos datos en general podemos determinar que el metodo funciona."


