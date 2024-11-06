import numpy as np

# ----- DEFINICIÓN DE LAS CAPAS -----
tamañoEntrada = 20
tamañoCapaOculta = 5
tamañoSalida = 3


# ----- FUNCIÓN PARA GENERAR TRAYECTORIAS POR TIPO -----
def generarTrayectoriasMovimientoPorTipo(tipoMovimiento, cantidadEjemplos):
    listaTrayectorias = []
    for _ in range(cantidadEjemplos):
        trayectoria = []
        if tipoMovimiento == "lineal":
            for paso in range(10):
                punto = [paso / 5, paso / 5]
                trayectoria.append(punto)
        elif tipoMovimiento == "circular":
            radio = 5
            for paso in range(10):
                angulo = (2 * np.pi * paso) / 10
                x = (radio * np.cos(angulo)) / 5
                y = (radio * np.sin(angulo)) / 5
                punto = [x, y]
                trayectoria.append(punto)
        elif tipoMovimiento == "aleatorio":
            for paso in range(10):
                x = np.random.uniform(-5, 5) / 5
                y = np.random.uniform(-5, 5) / 5
                punto = [x, y]
                trayectoria.append(punto)
        listaTrayectorias.append(trayectoria)
    trayectoriasEnArray = np.array(listaTrayectorias)
    trayectoriasRedimensionadas = trayectoriasEnArray.reshape(cantidadEjemplos, 20)
    return trayectoriasRedimensionadas


# ----- GENERACIÓN DE DATOS -----
datosLinea = generarTrayectoriasMovimientoPorTipo("lineal", 100)
datosCirculo = generarTrayectoriasMovimientoPorTipo("circular", 100)
datosAleatorio = generarTrayectoriasMovimientoPorTipo("aleatorio", 100)


# ----- ETIQUETAS -----
etiquetasLineales = [0] * 100
etiquetasCirculares = [1] * 100
etiquetasAleatorias = [2] * 100
etiquetas = np.array(etiquetasLineales + etiquetasCirculares + etiquetasAleatorias)


# ----- FUNCIÓN PARA INICIALIZAR PESOS -----
def inicializarPesosAleatorios(tamañoEntrada, tamañoCapaOculta, tamañoSalida):
    pesosEntradaOculta = np.random.uniform(
        -1, 1, size=(tamañoEntrada, tamañoCapaOculta)
    )
    pesosOcultaSalida = np.random.uniform(-1, 1, size=(tamañoCapaOculta, tamañoSalida))
    return pesosEntradaOculta, pesosOcultaSalida


# ----- INICIALIZACIÓN DE PESOS -----
pesosEntradaOculta, pesosOcultaSalida = inicializarPesosAleatorios(
    tamañoEntrada, tamañoCapaOculta, tamañoSalida
)
umbralOculta = np.random.uniform(-1, 1, size=tamañoCapaOculta)
umbralSalida = np.random.uniform(-1, 1, size=tamañoSalida)


# ----- FUNCIÓN SIGMOIDE -----
def sigmoide(x):
    xLimitado = np.clip(x, -500, 500)
    exponenteNegativo = -xLimitado
    eulerElevado = np.exp(exponenteNegativo)
    resultado = 1 / (1 + eulerElevado)
    return resultado


# ----- FUNCIÓN SOFTMAX -----
def Softmax(puntajes):
    expPuntajes = np.exp(puntajes - np.max(puntajes))
    return expPuntajes / expPuntajes.sum()


# ----- FUNCIÓN PARA GENERAR EL PERCEPTRÓN -----
def Perceptron(entrada, pesosEntradaOculta, pesosOcultaSalida):
    activacionesOcultas = sigmoide(np.dot(entrada, pesosEntradaOculta) + umbralOculta)
    sumaPonderadaSalida = np.dot(activacionesOcultas, pesosOcultaSalida) + umbralSalida
    return Softmax(sumaPonderadaSalida)


# ----- FUNCIÓN PARA ENTRENAR EL PERCEPTRÓN -----
def EntrenarPerceptron(entradas, etiquetas, epocas=500, tasaAprendizaje=0.01):
    global pesosEntradaOculta, pesosOcultaSalida
    for _ in range(epocas):
        for entrada, etiqueta in zip(entradas, etiquetas):
            activacionesOcultas = sigmoide(
                np.dot(entrada, pesosEntradaOculta) + umbralOculta
            )
            sumaPonderadaSalida = (
                np.dot(activacionesOcultas, pesosOcultaSalida) + umbralSalida
            )
            probabilidadesSalida = Softmax(sumaPonderadaSalida)
            etiquetaOneHot = np.zeros(3)
            etiquetaOneHot[etiqueta] = 1
            errorSalida = probabilidadesSalida - etiquetaOneHot
            gradienteSalida = errorSalida
            pesosOcultaSalida -= tasaAprendizaje * np.outer(
                activacionesOcultas, gradienteSalida
            )
            for j in range(tamañoCapaOculta):
                gradienteOculta = gradienteSalida @ pesosOcultaSalida[j, :]
                pesosEntradaOculta[:, j] -= tasaAprendizaje * gradienteOculta * entrada


# ----- FUNCIÓN PARA EVALUAR LA PRECISIÓN -----
def evaluarPrecision(modelo, datosPrueba, etiquetasPrueba):
    contadorAciertos = 0
    for entrada, etiquetaReal in zip(datosPrueba, etiquetasPrueba):
        salidaDelModelo = modelo(entrada, pesosEntradaOculta, pesosOcultaSalida)
        clasePredicha = np.argmax(salidaDelModelo)
        if clasePredicha == etiquetaReal:
            contadorAciertos += 1
    precision = contadorAciertos / len(etiquetasPrueba)
    return precision


# ----- FUNCIONES DETERMINAR LA PRECISIÓN POR TIPO -----
def evaluarPrecisionLineal():
    datosPrueba = generarTrayectoriasMovimientoPorTipo("lineal", 30)
    etiquetasPrueba = np.array([0] * 30)
    precision = evaluarPrecision(Perceptron, datosPrueba, etiquetasPrueba)
    return precision


def evaluarPrecisionCircular():
    datosPrueba = generarTrayectoriasMovimientoPorTipo("circular", 30)
    etiquetasPrueba = np.array([1] * 30)
    precision = evaluarPrecision(Perceptron, datosPrueba, etiquetasPrueba)
    return precision


def evaluarPrecisionAleatorio():
    datosPrueba = generarTrayectoriasMovimientoPorTipo("aleatorio", 30)
    etiquetasPrueba = np.array([2] * 30)
    precision = evaluarPrecision(Perceptron, datosPrueba, etiquetasPrueba)
    return precision


# ----- ENTRENAMIENTO DEL PERCEPTRÓN -----
EntrenarPerceptron(
    np.vstack((datosLinea, datosCirculo, datosAleatorio)), etiquetas, epocas=2500
)


# ----- IMPRIMIR RESULTADOS -----
precisionLineal = evaluarPrecisionLineal()
precisionCircular = evaluarPrecisionCircular()
precisionAleatorio = evaluarPrecisionAleatorio()

print(" -------- PRECISIÓN -------- ")
print(f"    Lineales    : {precisionLineal * 100:.2f}%")
print(f"    Circulares  : {precisionCircular * 100:.2f}%")
print(f"    Aleatorias  : {precisionAleatorio * 100:.2f}%")
