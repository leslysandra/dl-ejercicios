"""
02_ejemplo_xor_falla — CUANDO EL PERCEPTRÓN SIMPLE NO PUEDE APRENDER

Objetivo: mostrar la limitación más importante del
perceptrón simple, usando la compuerta lógica XOR (OR exclusivo).

Tabla de verdad de XOR:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0

A diferencia de AND y OR, XOR NO es linealmente separable: no existe
ninguna línea recta en el plano (x1, x2) que separe los puntos de
clase 1 de los de clase 0. Si dibujan los 4 puntos, van a ver que
las dos clases están "entrelazadas" en las esquinas del cuadrado.

Por eso, sin importar cuántas épocas le demos, el perceptrón simple
NUNCA va a converger a 0 errores en este dataset. Esta es la
limitación histórica real que casi mata la investigación en redes
neuronales en los años 70 (Minsky y Papert, 1969) — y la razón por
la que la Sesión 2 introduce el MLP (Perceptrón Multicapa): agregar
una capa oculta permite construir fronteras de decisión que ya no
son una sola línea recta.
"""

from perceptron import entrenar, predecir

datos_XOR = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

print("Intentando entrenar un perceptrón simple para aprender XOR...")
print("(dejamos correr muchas más épocas de lo normal, a propósito)\n")

w1, w2, b, historial = entrenar(
    datos_XOR,
    tasa_aprendizaje=0.1,
    epocas=50,   # muchas más de las que hicieron falta para AND/OR
    verbose=True,
)

print("\n--- Resultado final ---")
errores_finales = historial[-1]["errores"]
print(f"Errores en la última época: {errores_finales} de {len(datos_XOR)}")

if errores_finales > 0:
    print(
        "\nComo se esperaba: el perceptrón NO logró aprender XOR.\n"
        "Fíjense en la columna 'errores' de cada época: probablemente\n"
        "está oscilando (sube y baja) en vez de bajar de forma\n"
        "consistente hacia 0, como sí pasaba con AND y OR."
    )

print("\nPredicciones finales vs. esperadas:")
for x1, x2, y_real in datos_XOR:
    y_predicho = predecir(x1, x2, w1, w2, b)
    marca = "OK" if y_predicho == y_real else "FALLA"
    print(f"  {x1} XOR {x2} = {y_predicho} (esperado: {y_real}) [{marca}]")
