import numpy as np

# Definir las funciones del módulo perceptron.py aquí

def suma_ponderada(x1, x2, w1, w2, b):
    return x1 * w1 + x2 * w2 + b

def activacion_escalon(z):
    return 1 if z >= 0 else 0

def predecir(x1, x2, w1, w2, b):
    z = suma_ponderada(x1, x2, w1, w2, b)
    return activacion_escalon(z)

def entrenar(datos_entrenamiento, tasa_aprendizaje, epocas, verbose=False):
    # Inicialización de pesos y sesgo
    w1 = np.random.rand()
    w2 = np.random.rand()
    b = np.random.rand()

    historial_errores = []

    for epoca in range(epocas):
        total_error = 0
        for x1, x2, y_real in datos_entrenamiento:
            # Calcular la salida del perceptrón
            z = suma_ponderada(x1, x2, w1, w2, b)
            y_predicha = activacion_escalon(z)

            # Calcular el error
            error = y_real - y_predicha
            total_error += abs(error)

            # Actualizar pesos y sesgo
            w1 += tasa_aprendizaje * error * x1
            w2 += tasa_aprendizaje * error * x2
            b += tasa_aprendizaje * error
        
        historial_errores.append(total_error)
        if verbose:
            print(f"Época {epoca+1}: Error total = {total_error:.3f}, Pesos = ({w1:.3f}, {w2:.3f}), Sesgo = {b:.3f}")

    return w1, w2, b, historial_errores


"""
01_ejercicio_perceptron — EJEMPLO FUNDAMENTOS DE DL

Perceptrón simple que predice si un estudiante APRUEBA (1) o NO
APRUEBA (0), en base a dos datos de entrada:
    x1 = horas de estudio
    x2 = horas de práctica con ejercicios

La lógica matemática de la neurona (suma ponderada, activación,
regla de aprendizaje) está definida una sola vez en perceptron.py,
con comentarios detallados de cada fórmula. Este archivo se enfoca
en el caso de uso: preparar los datos, entrenar y probar.
"""

# ---------------------------------------------------------------------
# 1) LOS DATOS DE ENTRADA (dataset de juguete, inventado para la clase)
# ---------------------------------------------------------------------
# Cada elemento es (x1, x2, y):
#   x1 = horas de estudio, x2 = horas de práctica, y = aprobó (1/0)
#
# Este dataset es LINEALMENTE SEPARABLE a propósito: existe una línea
# recta en el plano (x1, x2) que separa perfectamente a los que
# aprueban de los que no. Esa es la condición que necesita un
# perceptrón simple para poder aprender el problema (ver 03_xor_falla.py
# para el caso en que esto NO se cumple).
datos_entrenamiento = [
    (1, 1, 0),
    (2, 1, 0),
    (2, 2, 0),
    (1, 3, 0),
    (3, 3, 1),
    (4, 3, 1),
    (4, 4, 1),
    (5, 5, 1),
]

# ---------------------------------------------------------------------
# 2) ENTRENAMIENTO
# ---------------------------------------------------------------------
print("--- ENTRENAMIENTO ---")
w1, w2, b, historial = entrenar(
    datos_entrenamiento,
    tasa_aprendizaje=0.1,
    epocas=10,
    verbose=True,
)

# ---------------------------------------------------------------------
# 3) PROBAR LA NEURONA YA ENTRENADA CON DATOS NUEVOS
# ---------------------------------------------------------------------
print("\n--- PRUEBA CON ESTUDIANTES NUEVOS (no vistos en el entrenamiento) ---")
estudiantes_nuevos = [
    (1, 2),  # pocas horas -> se espera 0
    (5, 4),  # muchas horas -> se espera 1
    (3, 1),  # caso intermedio, interesante para discutir en clase
]

for x1, x2 in estudiantes_nuevos:
    z = suma_ponderada(x1, x2, w1, w2, b)
    resultado = activacion_escalon(z)
    etiqueta = "APRUEBA" if resultado == 1 else "NO APRUEBA"
    print(f"Estudiante(horas_estudio={x1}, horas_practica={x2}) "
          f"-> z={z:.3f} -> {etiqueta}")
