"""
02_comparar_optimizadores.py — SGD vs SGD+MOMENTUM vs ADAM

Objetivo: hacer visible por qué Adam es hoy el optimizador por
defecto, comparando tres algoritmos sobre EXACTAMENTE el mismo
problema (XOR) y la misma arquitectura — la única variable que
cambia es el optimizador.

Idea clave: todos calculan el gradiente de la misma
forma (backpropagation), la diferencia está en CÓMO usan ese
gradiente para actualizar los pesos en cada paso.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from helpers import graficar_curvas

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")


def construir_modelo():
    """
    Misma arquitectura cada vez, para que la comparación entre
    optimizadores sea justa (no queremos que la diferencia de
    resultados venga de una arquitectura distinta).
    """
    return keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


# ---------------------------------------------------------------------
# Los tres optimizadores a comparar
# ---------------------------------------------------------------------
# SGD "puro": actualiza los pesos moviéndose en la dirección opuesta
# al gradiente, un paso de tamaño fijo (learning_rate) en cada
# iteración. Simple, pero puede ser lento y quedarse "zigzagueando"
# en superficies de error difíciles.
#
# SGD + momentum: agrega "inercia" al movimiento — si venimos
# moviéndonos en una dirección consistente, aceleramos en esa
# dirección en vez de reaccionar solo al gradiente del paso actual.
# Ayuda a atravesar zonas planas y a no oscilar tanto.
#
# Adam: combina la idea de momentum con una tasa de aprendizaje que
# se ADAPTA automáticamente por cada peso, según su historial de
# gradientes. Por eso suele converger más rápido y con menos ajuste
# manual de hiperparámetros — es la razón de que sea el default hoy.

optimizadores = {
    "SGD": keras.optimizers.SGD(learning_rate=0.5),
    "SGD + momentum": keras.optimizers.SGD(learning_rate=0.5, momentum=0.9),
    "Adam": keras.optimizers.Adam(learning_rate=0.1),
}

historiales = {}

for nombre, optimizador in optimizadores.items():
    print(f"\n--- Entrenando con {nombre} ---")

    tf.random.set_seed(0)  # misma inicialización de pesos en los 3 casos
    np.random.seed(0)

    modelo = construir_modelo()
    modelo.compile(
        optimizer=optimizador,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    historial = modelo.fit(X, y, epochs=300, verbose=0, batch_size=4)
    historiales[nombre] = historial

    perdida_final = historial.history["loss"][-1]
    exactitud_final = historial.history["accuracy"][-1]
    print(f"  Pérdida final: {perdida_final:.4f} | "
          f"Exactitud final: {exactitud_final:.4f}")

# ---------------------------------------------------------------------
# Comparación visual
# ---------------------------------------------------------------------
graficar_curvas(
    historiales,
    metrica="loss",
    titulo="Comparación de optimizadores — misma red, mismo dataset",
    nombre_archivo="comparacion_optimizadores.png",
)

