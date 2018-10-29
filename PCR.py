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

matriz=([[colum[0],colum[1],colum[2],colum[3],colum[4],colum[5],colum[6],colum[7],colum[8],colum[9]],[colum[10],colum[11],colum[12],colum[13],colum[14],colum[15],colum[16],colum[17],colum[18],colum[19]],[colum[20],colum[21],colum[22],colum[23],colum[24],colum[25],colum[26],colum[27],colum[28],colum[29]],[colum[30],colum[31],colum[32],colum[33],colum[34],colum[35],colum[36],colum[37],colum[38],colum[39]],[colum[40],colum[41],colum[42],colum[43],colum[44],colum[45],colum[46],colum[47],colum[48],colum[49]],[colum[50],colum[51],colum[52],colum[53],colum[54],colum[55],colum[56],colum[57],colum[58],colum[59]],[colum[60],colum[61],colum[62],colum[63],colum[64],colum[65],colum[66],colum[67],colum[68],colum[69]],[colum[70],colum[71],colum[72],colum[73],colum[74],colum[75],colum[76],colum[77],colum[78],colum[79]],[colum[80],colum[81],colum[82],colum[83],colum[84],colum[85],colum[86],colum[87],colum[88],colum[89]],[colum[90],colum[91],colum[92],colum[93],colum[94],colum[95],colum[96],colum[97],colum[98],colum[99]]])

print "La matriz de covarianza es:"
print matriz

#sacar autovectores y autovalores

x,y= np.linalg.eig(matriz)
print "radio"
print "Autovecotor",x[0]," Autovalores", y[0]
print "Textura" 
print "Autovecotor",x[1]," Autovalores", y[1]
print "Perimetro"
print"Autovecotor",x[2]," Autovalores", y[2]
print "Area"
print"Autovecotor",x[3]," Autovalores", y[3]
print "Smoothness"
print"Autovecotor",x[4]," Autovalores", y[4]
print "Compactness"
print"Autovecotor",x[5]," Autovalores", y[5]
print "Cancavidad"
print"Autovecotor",x[6]," Autovalores", y[6]
print "Puntos concavos"
print"Autovecotor",x[7]," Autovalores", y[7]
print "Simetria"
print"Autovecotor",x[8]," Autovalores", y[8]
print "Fractal dimension"
print"Autovecotor",x[9]," Autovalores", y[9]

print" "
print"Los parametros mas mportantes son el radio (",x[0], ")  y la textura (", x[1], ") debido a que tienen el mayor valor."
