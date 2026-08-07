"""
03_transfer_learning.py — TRANSFER LEARNING

Objetivo: en vez de entrenar una CNN desde pesos aleatorios (como en
02_cnn_desde_cero.py), partimos de un modelo YA entrenado sobre
millones de imágenes (MobileNetV2, entrenado sobre ImageNet) y
reutilizamos todo lo que ya "sabe ver" (bordes, texturas, formas,
patrones) — solo entrenamos una capa clasificadora nueva encima,
adaptada a NUESTRO problema específico.

Esta es la técnica que la mayoría de ustedes va a usar en el
proyecto final si su problema es de visión por computadora: con
pocos cientos (a veces decenas) de imágenes propias, suele superar
ampliamente a entrenar una CNN desde cero.

------------------------------------------------------------------
CÓMO ADAPTAR ESTE ARCHIVO A TU PROPIO PROYECTO
------------------------------------------------------------------
Este archivo viene configurado con un dataset de ejemplo (fotos de
flores, 5 clases) para que corra "tal cual" la primera vez. Para
usar TUS PROPIAS imágenes:

  1. Organizá tus imágenes en carpetas, una por clase:
        mis_datos/
            clase_a/
                imagen1.jpg
                imagen2.jpg
            clase_b/
                imagen1.jpg
                ...

  2. Cambiá la variable DIRECTORIO_DATOS más abajo para que apunte
     a la carpeta "mis_datos" (subida a Colab o a Google Drive).

  3. Corré el resto del archivo sin cambiar nada más.
------------------------------------------------------------------

Requiere conexión a internet la primera vez (descarga los pesos
preentrenados de MobileNetV2 y, si usás el dataset de ejemplo, las
fotos de flores).
"""

import time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(0)

# ---------------------------------------------------------------------
# CONFIGURACIÓN — lo único que necesitás tocar para usar tus datos
# ---------------------------------------------------------------------
# Por defecto, descarga un dataset de ejemplo (fotos de flores, 5
# clases). Reemplazá esta línea por la ruta a tu propia carpeta de
# imágenes cuando trabajes en tu proyecto.
directorio_ejemplo = keras.utils.get_file(
    "flower_photos",
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True,
)
DIRECTORIO_DATOS = directorio_ejemplo   # <-- TODO: cambiar por tu carpeta propia

TAMANO_IMAGEN = (160, 160)   # tamaño esperado por MobileNetV2 en esta config
TAMANO_BATCH = 32

# ---------------------------------------------------------------------
# 1) CARGAR LOS DATOS DESDE CARPETAS
# ---------------------------------------------------------------------
# image_dataset_from_directory arma el dataset automáticamente a
# partir de la estructura de carpetas (una subcarpeta = una clase).
dataset_entrenamiento = keras.utils.image_dataset_from_directory(
    DIRECTORIO_DATOS,
    validation_split=0.2,
    subset="training",
    seed=0,
    image_size=TAMANO_IMAGEN,
    batch_size=TAMANO_BATCH,
)

dataset_validacion = keras.utils.image_dataset_from_directory(
    DIRECTORIO_DATOS,
    validation_split=0.2,
    subset="validation",
    seed=0,
    image_size=TAMANO_IMAGEN,
    batch_size=TAMANO_BATCH,
)

nombres_clases = dataset_entrenamiento.class_names
numero_de_clases = len(nombres_clases)
print(f"Clases detectadas: {nombres_clases}")

# Optimización de rendimiento: mantiene lotes preparados en memoria
# mientras se entrena, para no esperar disco en cada paso.
dataset_entrenamiento = dataset_entrenamiento.prefetch(tf.data.AUTOTUNE)
dataset_validacion = dataset_validacion.prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------
# 2) EL MODELO BASE PREENTRENADO
# ---------------------------------------------------------------------
# include_top=False: descartamos la última capa de MobileNetV2
# (la que clasificaba entre las 1000 clases originales de ImageNet),
# porque nuestras clases son distintas.
#
# weights="imagenet": cargamos los pesos ya entrenados, en vez de
# empezar de pesos aleatorios como en 02_cnn_desde_cero.py.
modelo_base = keras.applications.MobileNetV2(
    input_shape=TAMANO_IMAGEN + (3,),
    include_top=False,
    weights="imagenet",
)

# Este es el paso conceptualmente más importante: CONGELAMOS el
# modelo base. Sus pesos NO se van a actualizar durante el
# entrenamiento — solo vamos a entrenar la capa clasificadora nueva
# que agregamos abajo. Por eso el entrenamiento es mucho más rápido
# y necesita muchos menos datos que entrenar todo desde cero.
modelo_base.trainable = False

# ---------------------------------------------------------------------
# 3) PREPROCESAMIENTO ESPERADO POR MOBILENETV2
# ---------------------------------------------------------------------
# Cada modelo preentrenado espera sus píxeles normalizados de una
# forma específica (distinta a la normalización simple /255.0 que
# usamos en 02_cnn_desde_cero.py). Usamos la función oficial del
# modelo para no equivocarnos.
preprocesar_entrada = keras.applications.mobilenet_v2.preprocess_input

# ---------------------------------------------------------------------
# 4) ARMAR EL MODELO COMPLETO: base congelada + clasificador nuevo
# ---------------------------------------------------------------------
entradas = keras.Input(shape=TAMANO_IMAGEN + (3,))
x = preprocesar_entrada(entradas)
x = modelo_base(x, training=False)  # training=False: importante para
                                      # que BatchNorm dentro del modelo
                                      # base se comporte en modo inferencia,
                                      # no de entrenamiento
x = keras.layers.GlobalAveragePooling2D()(x)   # resume cada mapa de
                                                  # características a
                                                  # un solo número
x = keras.layers.Dropout(0.2)(x)
salidas = keras.layers.Dense(numero_de_clases, activation="softmax")(x)

modelo = keras.Model(entradas, salidas)

modelo.summary()
print(f"\nParámetros entrenables: solo los de la capa clasificadora nueva "
      f"(el resto del modelo, {modelo_base.count_params():,} parámetros, "
      f"queda congelado).")

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------
# 5) ENTRENAMIENTO
# ---------------------------------------------------------------------
print("\n--- ENTRENAMIENTO (partiendo de un modelo ya entrenado) ---")
inicio = time.time()

historial = modelo.fit(
    dataset_entrenamiento,
    validation_data=dataset_validacion,
    epochs=5,
)

duracion = time.time() - inicio
print(f"\nTiempo de entrenamiento: {duracion:.1f} segundos")