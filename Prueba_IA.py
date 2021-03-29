import numpy as np

class RedNeuronal():
    
    def __init__(self):
        # Generacion de semilla para los numeros aleatorios
        np.random.seed(1)
        
        #Convertimos los pesos en una matriz de 3x1 con valores de -1 a 1 y con una media de 0
        self.pesos_sinapticos = 2 * np.random.random((3, 1)) - 1

    def sigmoide(self, x):
        #Aplicamos la funcion sigmoide
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoide(self, x):
        #Calcula la derivada de la funcion sigmoide
        return x * (1 - x)

    def entrenamiento(self, entradas_entrenamiento, salidas_entrenamiento, iteraciones):
        
        #Entrenamos el modelo para hacer precciones mientras ajusta los pesos
        for i in range(iteraciones):
            #Pasamos los datos de entrenamiento a traves de la neurona
            salida = self.pensar(entradas_entrenamiento)

            #Calculo de errores para propagacion hacia atras
            error = salidas_entrenamiento - salida
            
            #Ajustamos los pesos a traves de los errores
            ajustes = np.dot(entradas_entrenamiento.T, error * self.derivada_sigmoide(salida))

            self.pesos_sinapticos += ajustes

    def pensar(self, entradas):
        #Pasamos los datos de entrada para obtener la salida   
        #Hay que convertir la entrada en float
        
        entradas = entradas.astype(float)
        salida = self.sigmoide(np.dot(entradas, self.pesos_sinapticos))
        return salida


if __name__ == "__main__":

    #Inicializamos la clase neurona
    red_neuronal = RedNeuronal()

    print("Generando aleatoriamente los pesos: ")
    print(red_neuronal.pesos_sinapticos)

    #Seteamos los datos para el entrenamiento inicial
    entradas_entrenamiento = np.array([[0,0,1],
                                    [1,1,1],
                                    [0,0,0],
                                    [0,1,1]])

    salidas_entrenamiento = np.array([[0,1,0,1]]).T

    #Iniciamos el entrenamiento
    red_neuronal.entrenamiento(entradas_entrenamiento, salidas_entrenamiento, 15000)

    print("Pesos finales despues del entrenamiento: ")
    print(red_neuronal.pesos_sinapticos)

    primera_entrada = str(input("Introduzca la primera entrada: "))
    segunda_entrada = str(input("Introduzca la segunda entrada: "))
    tercera_entrada = str(input("Introduzca la tercera entrada: "))
    
    print("Considerando la nueva entrada: ", primera_entrada, segunda_entrada, tercera_entrada)
    print("El resultado seria: ")
    print(red_neuronal.pensar(np.array([primera_entrada, segunda_entrada, tercera_entrada])))
