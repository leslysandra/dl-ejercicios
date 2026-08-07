"""
03_activaciones_plantilla.py — EJERCICIO

Objetivo: comprobar con sus propias manos por qué la elección de
la función de activación en la capa oculta afecta qué tan rápido
(o si) el modelo aprende.

Van a entrenar EXACTAMENTE el mismo modelo tres veces, cambiando
solo la activación de la capa oculta: sigmoid, tanh y relu.

INSTRUCCIONES: completar las partes marcadas con TODO.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from helpers import graficar_curvas

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")

# ---------------------------------------------------------------------
# TODO 1: completar la lista con las tres activaciones a comparar.
# Los nombres válidos en Keras son strings: "sigmoid", "tanh", "relu"
# ---------------------------------------------------------------------
activaciones_a_probar = [
    "sigmoid",
    # TODO: agregar "tanh"
    # TODO: agregar "relu"
]

historiales = {}

for nombre_activacion in activaciones_a_probar:
    print(f"\n--- Entrenando con activación: {nombre_activacion} ---")

    tf.random.set_seed(0)
    np.random.seed(0)

    # TODO 2: completar la arquitectura del modelo. Debe tener:
    #   - una capa Input(shape=(2,))
    #   - una capa Dense oculta de 8 neuronas, usando
    #     `activation=nombre_activacion` (la variable del loop, no un
    #     string fijo — así se reusa el mismo código para las 3 pruebas)
    #   - una capa Dense de salida de 1 neurona, activation="sigmoid"
    modelo = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        # TODO: capa oculta
        # TODO: capa de salida
    ])

    # TODO 3: compilar el modelo (optimizer="adam",
    # loss="binary_crossentropy", metrics=["accuracy"])

    # TODO 4: entrenar con modelo.fit(...), guardando el resultado
    # en la variable `historial` (epochs=300, verbose=0, batch_size=4)
    historial = None  # reemplazar por la llamada real a fit()

    if historial is not None:
        historiales[nombre_activacion] = historial
        perdida_final = historial.history["loss"][-1]
        exactitud_final = historial.history["accuracy"][-1]
        print(f"  Pérdida final: {perdida_final:.4f} | "
              f"Exactitud final: {exactitud_final:.4f}")

# TODO 5: una vez completado lo anterior, descomentar para graficar
# graficar_curvas(
#     historiales,
#     metrica="loss",
#     titulo="Comparación de funciones de activación",
#     nombre_archivo="comparacion_activaciones.png",
# )

