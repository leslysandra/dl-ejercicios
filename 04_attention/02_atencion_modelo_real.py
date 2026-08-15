"""
02_atencion_modelo_real.py — ATENCIÓN DE UN MODELO YA ENTRENADO

Objetivo: repetir exactamente la misma idea de 01_self_attention_numpy.py
(extraer la matriz de atención de una oración), pero ahora sobre un
modelo Transformer REAL, entrenado sobre millones de textos — para
ver si aparecen los patrones lingüísticos que en el archivo anterior
no podían aparecer (porque ahí los pesos eran aleatorios).

Usamos Hugging Face `transformers`, que ya trae implementado todo lo
que programamos a mano en 01: Q, K, V, escalado, softmax, multi-head.
Acá no lo volvemos a programar — lo usamos, y nos quedamos con los
pesos de atención que el modelo calcula internamente.

Requiere conexión a internet la primera vez que se corre (descarga
el modelo preentrenado, unos cientos de MB). Requiere las librerías
`transformers` y `torch`:
    pip install transformers torch
"""

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------
# 1) CARGAR UN MODELO PEQUEÑO, YA ENTRENADO
# ---------------------------------------------------------------------
# DistilBERT multilingüe: una versión "destilada" (más chica y rápida)
# de BERT, entrenada en más de 100 idiomas incluido el español.
NOMBRE_MODELO = "distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)

# output_attentions=True: le pedimos al modelo que nos devuelva los
# pesos de atención de todas sus capas, no solo el resultado final.
# attn_implementation="eager": necesario en versiones recientes de
# transformers para poder extraer esos pesos (la implementación
# optimizada por defecto no los expone).
modelo = AutoModel.from_pretrained(
    NOMBRE_MODELO,
    output_attentions=True,
    attn_implementation="eager",
)
modelo.eval()  # modo evaluación: apaga dropout y similares

# ---------------------------------------------------------------------
# 2) LA ORACIÓN (la misma idea que en la Sesión 3 y en 01)
# ---------------------------------------------------------------------
oracion = "El gato que estaba en el tejado bajó"

# El tokenizer NO separa por espacios como hicimos nosotros a mano:
# separa en "subpalabras" (wordpieces), y agrega tokens especiales
# [CLS] al principio y [SEP] al final. Por eso la cantidad de tokens
# no siempre coincide con la cantidad de palabras.
entradas = tokenizer(oracion, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(entradas["input_ids"][0])

print(f"Oraci\u00f3n original: {oracion}")
print(f"Tokens reales que ve el modelo: {tokens}")

# ---------------------------------------------------------------------
# 3) FORWARD PASS: obtener las atenciones de todas las capas
# ---------------------------------------------------------------------
with torch.no_grad():  # no necesitamos gradientes, solo inferencia
    salida = modelo(**entradas, output_attentions=True)

# salida.attentions es una tupla: una posición por cada capa del
# modelo. Cada elemento tiene forma (batch, cabezas, tokens, tokens).
n_capas = len(salida.attentions)
n_cabezas = salida.attentions[0].shape[1]
print(f"\nEl modelo tiene {n_capas} capas, cada una con {n_cabezas} "
      f"cabezas de atenci\u00f3n (multi-head, como vimos en 01).")

# ---------------------------------------------------------------------
# 4) ELEGIR UNA CAPA Y UNA CABEZA PARA VISUALIZAR
# ---------------------------------------------------------------------
# Distintas capas/cabezas suelen capturar distintos tipos de
# relaciones. No hay una única "correcta" para mirar — parte del
# ejercicio es probar varias y comparar.
CAPA = 0     # 0 = primera capa (más cercana a la entrada)
CABEZA = 0   # 0 = primera cabeza de esa capa

matriz_atencion = salida.attentions[CAPA][0, CABEZA].numpy()
# forma: (n_tokens, n_tokens)

print(f"\nMatriz de atenci\u00f3n (capa {CAPA}, cabeza {CABEZA}):")
print("Cada fila suma ~1, igual que en el archivo anterior:")
print(matriz_atencion.sum(axis=-1).round(3))

# ---------------------------------------------------------------------
# 5) VISUALIZACIÓN
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
im = ax.imshow(matriz_atencion, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha="right")
ax.set_yticklabels(tokens)
ax.set_xlabel("Key (a quién se atiende)")
ax.set_ylabel("Query (quién pregunta)")
ax.set_title(f"Atención real — capa {CAPA}, cabeza {CABEZA}\n({NOMBRE_MODELO})")

plt.colorbar(im, ax=ax, label="peso de atención")
plt.tight_layout()
plt.savefig("atencion_modelo_real.png", dpi=120, bbox_inches="tight")
print("\nGr\u00e1fico guardado como atencion_modelo_real.png")
plt.show()