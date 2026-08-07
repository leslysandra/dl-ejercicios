"""
01_convolucion_visual.py — QUÉ HACE UNA CONVOLUCIÓN, VISUALMENTE

Objetivo: antes de entrenar cualquier CNN, ver con los propios ojos
qué es literalmente una convolución: una ventana chica (el "filtro"
o "kernel") que se desliza sobre la imagen, y en cada posición hace
una suma ponderada de los píxeles que tiene debajo — el mismo tipo
de operación (suma ponderada) que vimos en la neurona de la Sesión 1,
aplicada ahora sobre una vecindad de píxeles en vez de sobre todo
el input de una vez.

No usamos ningún dataset externo ni ningún framework de DL en este
archivo: generamos una imagen sintética simple con numpy y aplicamos
la convolución "a mano", con bucles for, para que cada paso sea
trazable. (En 02 y 03 sí usamos Keras, que hace esto mismo de forma
optimizada por dentro.)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1) UNA IMAGEN SINTÉTICA SIMPLE (nada de descargar datasets)
# ---------------------------------------------------------------------
# Una imagen en escala de grises es, matemáticamente, solo una matriz
# de números (0 = negro, 1 = blanco). Creamos una imagen con un
# cuadrado brillante en el centro, para que los bordes del cuadrado
# sean fáciles de reconocer visualmente en el resultado del filtro.
def crear_imagen_sintetica(tamano=28):
    imagen = np.zeros((tamano, tamano))
    margen = tamano // 4
    imagen[margen:-margen, margen:-margen] = 1.0
    return imagen


imagen = crear_imagen_sintetica()

# ---------------------------------------------------------------------
# 2) LA CONVOLUCIÓN, IMPLEMENTADA A MANO
# ---------------------------------------------------------------------
def convolucionar(imagen, kernel):
    """
    Desliza `kernel` (una matriz chica, ej. 3x3) sobre `imagen`,
    calculando en cada posición la suma ponderada de los píxeles
    cubiertos por el kernel. Esto es EXACTAMENTE lo que hace una
    capa Conv2D de Keras, solo que acá lo escribimos explícito.

    No aplicamos padding: la imagen de salida es un poco más chica
    que la de entrada (se "pierden" los bordes).
    """
    alto_img, ancho_img = imagen.shape
    alto_k, ancho_k = kernel.shape

    alto_salida = alto_img - alto_k + 1
    ancho_salida = ancho_img - ancho_k + 1
    salida = np.zeros((alto_salida, ancho_salida))

    for i in range(alto_salida):
        for j in range(ancho_salida):
            region = imagen[i:i + alto_k, j:j + ancho_k]
            # La operación central de la convolución: multiplicar
            # elemento a elemento y sumar todo — una suma ponderada,
            # igual que en la neurona del perceptrón.
            salida[i, j] = np.sum(region * kernel)

    return salida


# ---------------------------------------------------------------------
# 3) TRES FILTROS (KERNELS) CLÁSICOS, CON PESOS FIJOS
# ---------------------------------------------------------------------
# Estos pesos NO se aprenden acá (son valores conocidos, diseñados
# a mano hace décadas para tareas específicas de procesamiento de
# imágenes). En una CNN real, la red APRENDE estos pesos durante el
# entrenamiento — pero la operación que hace con ellos es esta misma.

kernel_borde_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])  # resalta cambios de intensidad de izquierda a derecha

kernel_borde_horizontal = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
])  # resalta cambios de intensidad de arriba a abajo

kernel_desenfoque = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]) / 9.0  # promedia la vecindad -> suaviza la imagen (blur)

filtros = {
    "Original": None,
    "Bordes verticales": kernel_borde_vertical,
    "Bordes horizontales": kernel_borde_horizontal,
    "Desenfoque (blur)": kernel_desenfoque,
}

# ---------------------------------------------------------------------
# 4) APLICAR Y VISUALIZAR
# ---------------------------------------------------------------------
fig, ejes = plt.subplots(1, len(filtros), figsize=(4 * len(filtros), 4))

for ax, (nombre, kernel) in zip(ejes, filtros.items()):
    if kernel is None:
        resultado = imagen
    else:
        resultado = convolucionar(imagen, kernel)

    ax.imshow(resultado, cmap="gray")
    ax.set_title(nombre)
    ax.axis("off")

plt.tight_layout()
plt.savefig("filtros_convolucion.png", dpi=120, bbox_inches="tight")
print("Gráfico guardado como filtros_convolucion.png")
plt.show()
