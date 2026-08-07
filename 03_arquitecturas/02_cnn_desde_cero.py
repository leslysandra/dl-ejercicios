"""
02_cnn_desde_cero.py — UNA CNN ENTRENADA DE PUNTA A PUNTA

Objetivo: entrenar una red convolucional completa, desde pesos
aleatorios, sobre un problema real de clasificación de imágenes
(Fashion-MNIST: 10 categorías de ropa, 28x28 píxeles, escala de
grises). Es un dataset chico a propósito, para que entrene en
poco tiempo en clase.

Este ejemplo cumple un rol específico en la sesión: mostrar que
"desde cero" SÍ funciona, pero notar cuántos datos y cuánto tiempo
necesitó — para que el contraste con transfer learning (03) se
sienta, no solo se explique.

Requiere conexión a internet la primera vez que se corre (Keras
descarga el dataset automáticamente y lo cachea localmente).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------
# 1) LOS DATOS: Fashion-MNIST
# ---------------------------------------------------------------------
# 60,000 imágenes de entrenamiento + 10,000 de prueba, 10 clases de
# prendas de ropa (camiseta, pantalón, zapato, etc.), 28x28 píxeles
# en escala de grises.
(X_entrenamiento, y_entrenamiento), (X_prueba, y_prueba) = \
    keras.datasets.fashion_mnist.load_data()

nombres_clases = [
    "Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]

print(f"Imágenes de entrenamiento: {X_entrenamiento.shape}")
print(f"Imágenes de prueba: {X_prueba.shape}")

# ---------------------------------------------------------------------
# 2) PREPROCESAMIENTO
# ---------------------------------------------------------------------
# Los píxeles vienen en el rango [0, 255]. Los escalamos a [0, 1]
# (normalización) porque las redes entrenan mejor y más rápido con
# valores de entrada chicos y en un rango consistente.
X_entrenamiento = X_entrenamiento.astype("float32") / 255.0
X_prueba = X_prueba.astype("float32") / 255.0

# Conv2D espera un canal de color explícito, aunque sea escala de
# grises (1 canal): forma (alto, ancho, canales)
X_entrenamiento = X_entrenamiento[..., np.newaxis]
X_prueba = X_prueba[..., np.newaxis]

# Para que la clase no espere 60,000 imágenes completas, usamos un
# subconjunto más chico — suficiente para ver el patrón de
# aprendizaje sin quemar 10 minutos de clase entrenando.
N_ENTRENAMIENTO = 6000
X_entrenamiento = X_entrenamiento[:N_ENTRENAMIENTO]
y_entrenamiento = y_entrenamiento[:N_ENTRENAMIENTO]

# ---------------------------------------------------------------------
# 3) LA ARQUITECTURA CNN
# ---------------------------------------------------------------------
modelo = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    # Primer bloque convolucional: 32 filtros de 3x3. La red va a
    # APRENDER estos filtros (a diferencia de 01_convolucion_visual.py,
    # donde los diseñamos a mano).
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    # MaxPooling reduce el tamaño espacial, quedándose con el valor
    # más alto de cada ventana — conserva "lo más activado" y
    # reduce la cantidad de cómputo de las capas siguientes.
    keras.layers.MaxPooling2D((2, 2)),

    # Segundo bloque: más filtros (64), buscando patrones más
    # complejos construidos sobre los patrones simples del primer
    # bloque (bordes -> combinaciones de bordes -> formas).
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    # Aplanar el mapa de características 2D a un vector 1D, para
    # poder pasarlo a capas Dense — el mismo tipo de capa que
    # usamos en el MLP de la Sesión 2.
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),  # apaga aleatoriamente 30% de las
                                  # neuronas en cada paso de entrenamiento,
                                  # para reducir sobreajuste (overfitting)

    # Capa de salida: 10 neuronas (una por clase), softmax convierte
    # las salidas en probabilidades que suman 1 entre las 10 clases.
    keras.layers.Dense(10, activation="softmax"),
])

modelo.summary()

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # como MNIST tiene 10
                                               # clases (no binario),
                                               # usamos esta variante
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------
# 4) ENTRENAMIENTO
# ---------------------------------------------------------------------
print("\n--- ENTRENAMIENTO (desde pesos aleatorios) ---")
import time
inicio = time.time()

historial = modelo.fit(
    X_entrenamiento, y_entrenamiento,
    epochs=5,
    validation_split=0.1,  # separa 10% del set de entrenamiento
                            # para vigilar overfitting en vivo
    verbose=1,
)

duracion = time.time() - inicio
print(f"\nTiempo de entrenamiento: {duracion:.1f} segundos "
      f"para {N_ENTRENAMIENTO} imágenes, 5 épocas")

# ---------------------------------------------------------------------
# 5) EVALUACIÓN SOBRE DATOS NUNCA VISTOS
# ---------------------------------------------------------------------
perdida_prueba, exactitud_prueba = modelo.evaluate(X_prueba, y_prueba, verbose=0)
print(f"\nExactitud sobre el conjunto de prueba: {exactitud_prueba:.3f}")
