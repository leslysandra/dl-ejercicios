"""
helpers.py — Funciones auxiliares compartidas para esta carpeta

Todo lo relacionado a graficar curvas de entrenamiento vive acá,
para no repetir el mismo código de matplotlib en cada ejercicio.

    from helpers import graficar_curvas
"""

import matplotlib.pyplot as plt


def graficar_curvas(historiales, metrica="loss", titulo="Comparación",
                     nombre_archivo=None):
    """
    Grafica una o varias curvas de entrenamiento en el mismo eje,
    para compararlas visualmente.

    historiales: dict donde cada clave es una etiqueta (ej. "Adam")
                 y cada valor es un objeto History de Keras
                 (lo que devuelve model.fit(...)).
    metrica:     qué métrica graficar, ej. "loss" o "accuracy".
    """
    plt.figure(figsize=(8, 5))

    for etiqueta, historial in historiales.items():
        valores = historial.history[metrica]
        plt.plot(valores, label=etiqueta, linewidth=2)

    plt.xlabel("Época")
    plt.ylabel(metrica)
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=120, bbox_inches="tight")
        print(f"Gráfico guardado como {nombre_archivo}")

    plt.show()
