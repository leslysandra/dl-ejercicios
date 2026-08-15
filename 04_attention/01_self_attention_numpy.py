"""
01_self_attention_numpy.py — SELF-ATTENTION DESDE CERO, SOLO NUMPY

Objetivo: programar con las manos el cálculo exacto.
No usamos ningún framework de deep learning: solo NumPy, para que
cada paso (Q, K, V, el producto punto, el escalado, softmax) sea
trazable línea por línea — el mismo espíritu que 01_perceptron/
en la Sesión 1.

AVISO IMPORTANTE (leer antes de correr el código):
Los "embeddings" y los pesos W_Q, W_K, W_V de este archivo son
ALEATORIOS, no vienen de ningún entrenamiento. Eso significa que
la matriz de atención que va a salir NO va a mostrar un patrón
lingüístico real (no va a "saber" que gato es el sujeto de bajó).
El objetivo de este archivo es entender el CÁLCULO, no interpretar
el resultado como si tuviera sentido semántico. Para ver atención
de un modelo YA entrenado, con patrones reales, ver el archivo
02_atencion_modelo_real.py.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------------------------------------------------------------
# 1) LA ORACIÓN Y SUS EMBEDDINGS (inventados, solo para el cálculo)
# ---------------------------------------------------------------------
# En un modelo real, cada palabra se convierte en un vector (embedding)
# aprendido durante el entrenamiento. Acá generamos vectores aleatorios
# de dimensión chica (d_model=8) solo para tener algo con qué calcular.
tokens = ["el", "gato", "duerme", "mucho"]
n_tokens = len(tokens)
d_model = 8   # dimensión de cada embedding (chica, para que sea legible)
d_k = 4       # dimensión de Q, K, V (en modelos reales, d_model se
              # divide entre varias "cabezas" de atención — ver el
              # comentario de multi-head más abajo)

# Cada fila de X es el embedding de un token: X.shape = (4, 8)
X = np.random.randn(n_tokens, d_model)

# ---------------------------------------------------------------------
# 2) LAS MATRICES DE PESOS: Q, K, V
# ---------------------------------------------------------------------
# La idea central de la atención es que cada palabra genera TRES
# versiones distintas de sí misma, mediante tres matrices de pesos
# que SÍ se aprenden durante el entrenamiento (acá las dejamos
# aleatorias, sin entrenar):
#
#   Query (Q):  "qué estoy buscando yo" — la pregunta que hace cada
#               palabra sobre el resto de la oración
#   Key (K):    "qué ofrezco yo" — cómo se presenta cada palabra
#               para que otras la puedan encontrar relevante
#   Value (V):  "qué informacion doy yo" — el contenido real que se
#               mezcla si termino siendo relevante
#
# Estas tres matrices convierten cada embedding (dimensión d_model)
# en un vector más chico (dimensión d_k).
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_k) * 0.5

Q = X @ W_Q   # (4, 4) — una "pregunta" por token
K = X @ W_K   # (4, 4) — una "llave" por token
V = X @ W_V   # (4, 4) — un "contenido" por token

print("Embeddings de entrada (X):")
print(np.round(X, 2))
print(f"\nQ, K, V tienen forma: {Q.shape} (una fila por token)")


# ---------------------------------------------------------------------
# 3) EL PRODUCTO PUNTO: ¿QUÉ TAN "COMPATIBLE" ES CADA PAR DE PALABRAS?
# ---------------------------------------------------------------------
# Para saber cuánto debe "atender" el token i al token j, calculamos
# el producto punto entre la Query del token i y la Key del token j.
# Un producto punto alto significa que esos dos vectores "apuntan en
# una dirección parecida" — es una medida de similitud/compatibilidad.
#
# Hacemos esto para TODOS los pares a la vez con una multiplicación
# de matrices: Q @ K.T
scores_crudos = Q @ K.T   # forma (4, 4): scores_crudos[i, j] = compatibilidad
                            # entre el token i (como query) y el token j (como key)

print("\nScores crudos (Q @ K.T), antes de escalar:")
print(np.round(scores_crudos, 2))

# ---------------------------------------------------------------------
# 4) ESCALADO: DIVIDIR POR raíz(d_k)
# ---------------------------------------------------------------------
# Sin este paso, cuando d_k es grande, los productos punto tienden a
# tener valores muy grandes, lo que empuja al softmax del siguiente
# paso a producir distribuciones casi "todo o nada" (gradientes muy
# chicos, difícil de entrenar). Dividir por raíz(d_k) mantiene los
# valores en un rango más manejable. Por esto la técnica se llama
# "scaled dot-product attention".
scores_escalados = scores_crudos / np.sqrt(d_k)

print(f"\nScores escalados (÷ raíz({d_k})={np.sqrt(d_k):.2f}):")
print(np.round(scores_escalados, 2))


# ---------------------------------------------------------------------
# 5) SOFTMAX: CONVERTIR LOS SCORES EN PESOS DE ATENCIÓN (SUMAN 1)
# ---------------------------------------------------------------------
def softmax(x):
    """
    Convierte una fila de números cualquiera en una distribución de
    probabilidad: todos positivos, y la fila suma exactamente 1.
    Restamos el máximo antes de exponenciar solo por estabilidad
    numérica (no cambia el resultado matemático).
    """
    x_estable = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_estable)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


pesos_atencion = softmax(scores_escalados)   # forma (4, 4)

print("\nMATRIZ DE ATENCIÓN (cada fila suma 1):")
print("      " + "  ".join(f"{t:>8}" for t in tokens))
for i, fila in enumerate(pesos_atencion):
    valores = "  ".join(f"{v:8.3f}" for v in fila)
    print(f"{tokens[i]:>6}{valores}")

print(
    "\nCada fila responde: 'cuando este token pregunta (Query), "
    "¿qué porcentaje de atención le da a cada palabra de la oración "
    "(incluida a sí mismo)?' — recuerden que con pesos aleatorios "
    "esto NO tiene un patrón lingüístico real todavía."
)


# ---------------------------------------------------------------------
# 6) LA SALIDA: MEZCLAR LOS VALUES SEGÚN LOS PESOS DE ATENCIÓN
# ---------------------------------------------------------------------
# El resultado final de la capa de atención, para cada token, es un
# promedio ponderado de los VALUES de toda la oración, usando los
# pesos de atención que acabamos de calcular.
salida = pesos_atencion @ V   # forma (4, 4)

print("\nSalida de la capa de atención (una fila por token, mezcla de V):")
print(np.round(salida, 2))


# ---------------------------------------------------------------------
# 7) VISUALIZACIÓN
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(pesos_atencion, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(range(n_tokens))
ax.set_yticks(range(n_tokens))
ax.set_xticklabels(tokens)
ax.set_yticklabels(tokens)
ax.set_xlabel("Key (a quién se atiende)")
ax.set_ylabel("Query (quién pregunta)")
ax.set_title("Matriz de atención (pesos sin entrenar)")

for i in range(n_tokens):
    for j in range(n_tokens):
        ax.text(j, i, f"{pesos_atencion[i, j]:.2f}", ha="center", va="center",
                 color="white" if pesos_atencion[i, j] > 0.5 else "black", fontsize=10)

plt.colorbar(im, ax=ax, label="peso de atención")
plt.tight_layout()
plt.savefig("atencion_numpy.png", dpi=120, bbox_inches="tight")
print("\nGráfico guardado como atencion_numpy.png")
plt.show()
