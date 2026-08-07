"""
01_mlp_resuelve_xor.py

El perceptrón simple FALLA en aprender XOR, porque XOR
no es linealmente separable y una sola neurona solo puede trazar
una línea recta como frontera de decisión.

Resolvemos exactamente ese problema con un MLP (Multilayer
Perceptron): una red con una CAPA OCULTA entre la entrada y la
salida. La capa oculta permite combinar varias rectas y así formar
fronteras de decisión más complejas — esto es, en esencia, lo que
dice el Universal Approximation Theorem: una red con suficientes
neuronas ocultas puede aproximar cualquier función razonable.

A partir de acá dejamos de programar la neurona "a mano" y usamos
Keras/TensorFlow, que implementa por nosotros la suma ponderada,
la activación y —lo nuevo de hoy— el algoritmo de backpropagation
para ajustar los pesos automáticamente.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Reproducibilidad: misma inicialización de pesos cada vez que se corre
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# 1) LOS DATOS DE ENTRADA — la misma tabla XOR de la Sesión 1
# ---------------------------------------------------------------------
# Keras espera arrays numéricos (numpy), no listas de tuplas como
# usábamos en el perceptrón puro.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype="float32")

y = np.array([
    [0],
    [1],
    [1],
    [0],
], dtype="float32")

# ---------------------------------------------------------------------
# 2) LA ARQUITECTURA: MLP CON UNA CAPA OCULTA
# ---------------------------------------------------------------------
# keras.Sequential apila capas una tras otra. Cada Dense es una capa
# "totalmente conectada": cada neurona recibe TODAS las salidas de
# la capa anterior, exactamente la misma matemática del perceptrón
# (suma ponderada + activación) pero ahora repetida en paralelo
# varias veces y en cascada.
modelo = keras.Sequential([
    keras.layers.Input(shape=(2,)),

    # Capa oculta: 8 neuronas. Cada una aprende su propia "línea recta"
    # en el espacio de entrada; combinadas, pueden formar una frontera
    # curva/quebrada como la que XOR necesita.
    # activation="relu": función de activación moderna (ver comentario
    # más abajo, sección 3).
    keras.layers.Dense(8, activation="relu"),

    # Capa de salida: 1 neurona con activación sigmoid, que aplasta
    # el resultado a un rango (0, 1), interpretable como probabilidad
    # de pertenecer a la clase 1.
    keras.layers.Dense(1, activation="sigmoid"),
])

modelo.summary()

# ---------------------------------------------------------------------
# 3) POR QUÉ RELU Y NO LA FUNCIÓN ESCALÓN
# ---------------------------------------------------------------------
# La función escalón (step function) no sirve
# para entrenar con gradientes: es plana en casi todos lados y tiene
# un salto brusco en 0, así que su derivada es 0 (o no existe) casi
# siempre. Backpropagation necesita derivadas útiles para saber en
# qué dirección mover cada peso.
#
# ReLU(z) = max(0, z) sí tiene una derivada útil (0 o 1) y es la
# activación más usada hoy en capas ocultas por ser simple y barata
# de calcular. Sigmoid se sigue usando en la capa de SALIDA cuando
# queremos interpretar el resultado como una probabilidad.

# ---------------------------------------------------------------------
# 4) COMPILAR: FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# ---------------------------------------------------------------------
# loss="binary_crossentropy": mide qué tan lejos está la probabilidad
# predicha de la etiqueta real (0 o 1). Es el reemplazo "moderno" del
# error simple (y_real - y_predicho) que usábamos en el perceptrón.
#
# optimizer="adam": el algoritmo que ajusta los pesos usando el
# gradiente del error (backpropagation) en cada paso. Adam es hoy
# el optimizador por defecto más usado — lo comparamos con otros
# en 02_comparar_optimizadores.py.
modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------
# 5) ENTRENAMIENTO
# ---------------------------------------------------------------------
# A diferencia del perceptrón (donde programamos el bucle de
# entrenamiento a mano), acá Keras hace todo el forward pass,
# backward pass (backpropagation) y ajuste de pesos con una sola
# llamada: model.fit(...)
print("\n--- ENTRENAMIENTO ---")
historial = modelo.fit(
    X, y,
    epochs=500,       # XOR con una red chica necesita bastantes épocas
    verbose=0,        # silenciamos el log línea por línea (son 500)
    batch_size=4,     # el dataset completo cabe en un solo batch
)

perdida_final = historial.history["loss"][-1]
exactitud_final = historial.history["accuracy"][-1]
print(f"Pérdida final: {perdida_final:.4f}")
print(f"Exactitud final: {exactitud_final:.4f}")

# ---------------------------------------------------------------------
# 6) PROBAR EL MODELO
# ---------------------------------------------------------------------
print("\n--- PREDICCIONES ---")
predicciones = modelo.predict(X, verbose=0)

for entrada, prediccion_prob, esperado in zip(X, predicciones, y):
    prediccion_clase = 1 if prediccion_prob[0] >= 0.5 else 0
    marca = "OK" if prediccion_clase == int(esperado[0]) else "FALLA"
    print(f"  {int(entrada[0])} XOR {int(entrada[1])} = {prediccion_clase} "
          f"(prob={prediccion_prob[0]:.3f}, esperado={int(esperado[0])}) [{marca}]")

print(
    "\nA diferencia del perceptrón simple (Sesión 1), que nunca bajaba\n"
    "de varios errores en XOR sin importar cuántas épocas le diéramos,\n"
    "este MLP con una sola capa oculta SÍ logra aprender el patrón."
)