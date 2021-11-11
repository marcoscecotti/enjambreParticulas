# pos actual de la particula xi(t)
# vel actual de la particula vi(t)
# mejor posicion historica de la particula yi(t)
# mejor posicion historica global o de su vecindario de la particula y^i(t)

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import time

class Particula:
    def __init__(self, posicion, velocidad):
        self.posicion = np.array(posicion)
        self.velocidad = np.array(velocidad)
        self.mejorLocal = np.array(posicion)


print("¿Que función desea representar? (a, b o c)")
ejercicio=input()

if ejercicio == "a":
    # Rango de particulas
    R = np.array([[-512,512]])
    def f(x):
        return -x*np.sin(np.sqrt(np.abs(x)))
    toleranciaFitness = -400 # Tolerancia de fitness
    x = np.linspace(R[0][0],R[0][1],(R[0][1]-R[0][0]),endpoint=True)

if ejercicio == "b":
    R = np.array([[0,20]]) 
    def f(x):
        return x + np.sin(3*x)+8*np.cos(5*x)
    toleranciaFitness = -5.5 # Tolerancia de fitness
    x = np.linspace(R[0][0],R[0][1],(R[0][1]-R[0][0])+100,endpoint=True)

if ejercicio == "c":
    R = np.array([[-100,100],[-100,100]])
    def f(x):
        return (sum(x**2))**0.25*((np.sin(50*(sum(x**2))**0.1))**2+1)
    
    def f_graf(x,y):
        return (x**2+y**2)**0.25*((np.sin(50*(x**2+y**2)**0.1))**2+1)

    toleranciaFitness = -0.5 # Tolerancia de fitness


n = 50          # Particulas
N = len(R)      # Dimensiones
E = 200         # Epocas
maxIteraciones = 1000 # Maximo numero de iteraciones
enjambre = []
particulaGanadora = 0

print("Dimensión: ", N)

t = time.time()

# Inicializamos las particulas aleatoreamente
for k in range(n): # Cada particula
    # Para cada dimension
    posicion = []
    for i in range(N):
        posicion.append(random.uniform(R[i][0], R[i][1]))
    enjambre.append(Particula(posicion, [0]*N)) # Particula inicializada

# Graficamos particulas iniciales
# for k in range(n): # Cada particula
#     plt.plot(enjambre[k].posicion, f(enjambre[k].posicion), 'bo', alpha=0.1)

mejorGlobal = enjambre[0].mejorLocal # Posicion inicial del mejor global
mejorFitness = f(mejorGlobal) # Fitness inicial del mejor global

i = 0
while (mejorFitness > toleranciaFitness).all() and (i < maxIteraciones):
    # Para cada particula
    for p, particula in enumerate(enjambre):
        # Si la posicion actual del individuo es mejor que su mejor local, entonces actualizo
        if (f(particula.posicion) < f(particula.mejorLocal)).all():
            particula.mejorLocal = particula.posicion
        # Si la mejor posicion histórica del individuo es mejor que la posición global de la bandada, entonces actualizo
        if (f(particula.mejorLocal) < f(mejorGlobal)).all():
            mejorGlobal = particula.mejorLocal
            mejorFitness = f(mejorGlobal)
            particulaGanadora = p
    
    for p, particula in enumerate(enjambre):
        # Para cada particula

        r1 = np.random.rand(2)
        r2 = np.random.rand(2)


        c1 = 2.5 - (2 / maxIteraciones) * i
        c2 = .5 - (2 / maxIteraciones)  * i
        
        #actualizo velocidad
        particula.velocidad = particula.velocidad + c1 * r1 * (particula.mejorLocal - particula.posicion) + c2 * r2 *(mejorGlobal - particula.posicion)    
        #actualizo posicion
        particula.posicion = particula.posicion +  particula.velocidad
        print(particula.posicion)
    
    # for k in range(n): # Cada particula
    #     plt.plot(enjambre[k].posicion, f(enjambre[k].posicion), 'bo', alpha=0.02)
    i += 1

print ("Mejor global: ", mejorGlobal)
print ("Mejor fitness: ", mejorFitness)
print ("Particula ganadora: ", enjambre[particulaGanadora].posicion)
#print("Tiempo: ",time.time() - t)

if ejercicio == "a" or ejercicio == "b":
    plt.plot(x,f(x))
    plt.title('Función a minimizar a')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(enjambre[particulaGanadora].posicion[0], mejorFitness[0],'o')

if ejercicio == "c":
    colorGraf = cm.get_cmap('coolwarm')
    #Graficar punto c
    x3 = np.linspace(R[0][0],R[0][1],100,endpoint=True)
    y3 = np.linspace(R[1][0],R[1][1],100,endpoint=True)

    fig = plt.figure()
    # ax = Axes3D(fig)
    x3, y3 = np.meshgrid(x3, y3)
    z3 = f_graf(x3,y3)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap=colorGraf)
    # plt.title('Proyección y minimo global')

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.title('Función objetivo')
    ax = fig.add_subplot(1, 2, 2)
    cset = ax.contourf(x3, y3, z3, zdir ='z', offset = np.min(z3), cmap = colorGraf) 
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.plot(enjambre[particulaGanadora].posicion[0],enjambre[particulaGanadora].posicion[1],mejorFitness,'o')

plt.show()