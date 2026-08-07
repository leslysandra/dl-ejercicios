"""
04_visualizacion_learning_rate.py — EL EFECTO DEL LEARNING RATE

Objetivo: ver el problema clásico del learning rate: si es muy chico, 
se aprende muy lento; si es muy grande, el entrenamiento se vuelve inestable (oscila) 
o incluso diverge (el error crece en vez de bajar).

Usamos SGD puro (no Adam) a propósito: Adam ajusta la tasa de
aprendizaje automáticamente por dentro, lo que disimula el efecto
que queremos mostrar acá.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from helpers import graficar_curvas

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")

# ---------------------------------------------------------------------
# Cuatro learning rates a comparar: muy chico, chico, razonable, grande
# ---------------------------------------------------------------------
learning_rates = [0.01, 0.1, 0.5, 3.0]

historiales = {}

for lr in learning_rates:
    etiqueta = f"lr = {lr}"
    print(f"\n--- Entrenando con {etiqueta} ---")

    tf.random.set_seed(0)
    np.random.seed(0)

    modelo = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    modelo.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    historial = modelo.fit(X, y, epochs=300, verbose=0, batch_size=4)
    historiales[etiqueta] = historial

    perdida_final = historial.history["loss"][-1]
    exactitud_final = historial.history["accuracy"][-1]
    print(f"  Pérdida final: {perdida_final:.4f} | "
          f"Exactitud final: {exactitud_final:.4f}")

graficar_curvas(
    historiales,
    metrica="loss",
    titulo="Efecto del learning rate (SGD puro, misma red)",
    nombre_archivo="comparacion_learning_rate.png",
)
