"""
visualizacion_frontera.py — VER LA FRONTERA DE DECISIÓN MOVERSE

Objetivo: hacer visible lo que hasta ahora solo veíamos como números
en la consola. Graficamos los puntos del dataset y la recta que separa las dos clases, en distintos momentos
del entrenamiento.

Nota sobre "sin frameworks": el CÁLCULO del perceptrón sigue siendo
100% Python puro (importado de perceptron.py). Acá usamos matplotlib
únicamente para DIBUJAR, no para entrenar ni predecir — es una
excepción justificada porque no existe una forma razonable de
graficar sin una librería de gráficos.

Requiere: pip install matplotlib
"""

import matplotlib.pyplot as plt
from perceptron import entrenar, pesos_iniciales_aleatorios

# Mismo dataset del ejemplo original (Sesión 1):
# (horas_estudio, horas_practica, aprobo)
datos = [
    (1, 1, 0),
    (2, 1, 0),
    (2, 2, 0),
    (1, 3, 0),
    (3, 3, 1),
    (4, 3, 1),
    (4, 4, 1),
    (5, 5, 1),
]


def graficar_frontera(ax, datos, w1, w2, b, titulo):
    """
    Dibuja los puntos del dataset y la recta de decisión w1*x1 + w2*x2 + b = 0
    despejada como: x2 = -(w1*x1 + b) / w2
    """
    # Puntos, coloreados por clase real
    for x1, x2, y in datos:
        color = "tab:green" if y == 1 else "tab:red"
        marcador = "o" if y == 1 else "x"
        ax.scatter(x1, x2, c=color, marker=marcador, s=100, zorder=3)

    # La recta de decisión (solo se puede dibujar si w2 != 0)
    if abs(w2) > 1e-6:
        x1_rango = [0, 6]
        x2_rango = [-(w1 * x + b) / w2 for x in x1_rango]
        ax.plot(x1_rango, x2_rango, "b--", linewidth=2, zorder=2)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel("horas de estudio (x1)")
    ax.set_ylabel("horas de práctica (x2)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)


# Entrenamos guardando el historial de pesos por época
w1_inicial, w2_inicial, b_inicial = pesos_iniciales_aleatorios(semilla=42)
w1_final, w2_final, b_final, historial = entrenar(
    datos,
    tasa_aprendizaje=0.1,
    epocas=10,
    w1=w1_inicial, w2=w2_inicial, b=b_inicial,
    verbose=False,
)

# Elegimos 4 momentos del entrenamiento para mostrar la evolución:
# antes de entrenar, dos puntos intermedios, y el resultado final.
indices_a_mostrar = sorted(set([
    0,
    len(historial) // 3,
    (2 * len(historial)) // 3,
    len(historial) - 1,
]))

fig, ejes = plt.subplots(1, len(indices_a_mostrar), figsize=(5 * len(indices_a_mostrar), 5))
if len(indices_a_mostrar) == 1:
    ejes = [ejes]

# Primer panel especial: pesos iniciales, antes de cualquier ajuste
graficar_frontera(
    ejes[0], datos, w1_inicial, w2_inicial, b_inicial,
    "Antes de entrenar (pesos aleatorios)"
)

for i, idx in enumerate(indices_a_mostrar[1:], start=1):
    paso = historial[idx]
    graficar_frontera(
        ejes[i], datos, paso["w1"], paso["w2"], paso["b"],
        f"Época {paso['epoca']} (errores={paso['errores']})"
    )

plt.tight_layout()
plt.savefig("evolucion_frontera_decision.png", dpi=120)
print("Gráfico guardado como evolucion_frontera_decision.png")
plt.show()
