# Implementacion austera de una red neuronal de 2 capas con comentarios en espanol.

# Numpy es una extension que introduce una variedad de funciones matematicas y facilita operar con vectores y matrices.
import numpy as np

'''
A continuacion definimos nuestra funcion de activacion no-lineal.
Para esto elegimos utilizar la funcion sigmoide debido a que:
- Esta acotada.
- Es facilmente derivable.
- Es monotona.

http://qr.ae/1jUNNH
'''

def sigmoide(x, deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


# <DATOS DE ENTRENAMIENTO>
'''
A continuacion estan los datos de entrenamiento. 
Se describen los datos de entrada y los esperados de salida.
'''
# Vectores de entrada
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# Valores de salida
y = np.array([[0,0,1,1]]).T

# </DATOS DE ENTRENAMIENTO>

'''
La libreria cuenta con un generador de numeros pseudoaleatorios.
Inicializamos el generador con una semilla predeterminada para
hacer el calculo deterministico.
''' 
np.random.seed(1)

'''
pesos0 es la primer capa de pesos que conecta la capa0 con la capa1.
pesos0 es una vector de dimension (3,1) con valores al azar con promedio 0.
'''
pesos0 = 2*np.random.random((3,1)) - 1

# Iteramos 1000 veces
for iter in xrange(1000):
    # Propagacion hacia atras o retropropagacion (Backpropagation)
    capa0 = X
    capa1 = sigmoide(np.dot(capa0, pesos0))

    # Cuanto error cometimos?
    capa1_error = y - capa1

    # Multiplicamos el error por la pendiende de la funcion sigmoide en los valores de capa1
    capa1_delta = capa1_error * sigmoide(capa1, True)

    # Actualizamos los pesos0
    pesos0 += np.dot(capa0.T, capa1_delta)

print "Resultado despues de entrenar:"
print capa1