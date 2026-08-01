"""
PERCEPTRON DESDE CERO — SIN FRAMEWORKS
Módulo Deep Learning

Objetivo: entender qué hace matemáticamente UNA sola neurona artificial
(un perceptrón) cuando toma una decisión de clasificación binaria.

Ejemplo elegido: predecir si un estudiante APRUEBA (1) o NO APRUEBA (0)
un examen, con base a dos datos de entrada (features):
    x1 = horas de estudio
    x2 = horas de práctica con ejercicios

No usamos numpy ni ninguna librería de ML: todo el cálculo se hace
"a mano" con listas y bucles for, para que cada línea sea trazable
hasta la fórmula matemática que representa.
"""

import random

# ---------------------------------------------------------------------
# 1) LOS DATOS DE ENTRADA (dataset de juguete, inventado para la clase)
# ---------------------------------------------------------------------
# Cada elemento es (x1, x2, y):
#   x1 = horas de estudio
#   x2 = horas de práctica
#   y  = etiqueta real (1 = aprobó, 0 = no aprobó)
#
# Este dataset es LINEALMENTE SEPARABLE a propósito: existe una línea
# recta en el plano (x1, x2) que separa perfectamente a los que
# aprueban de los que no. Esa es justamente la limitación clásica
# de un perceptrón simple: solo puede aprender fronteras de decisión
# que sean una línea recta (o un hiperplano, en más dimensiones).
datos_entrenamiento = [
    # (horas_estudio, horas_practica, aprobo)
    (1, 1, 0),
    (2, 1, 0),
    (2, 2, 0),
    (1, 3, 0),
    (3, 3, 1),
    (4, 3, 1),
    (4, 4, 1),
    (5, 5, 1),
]

# ---------------------------------------------------------------------
# 2) INICIALIZACIÓN DE LOS PARÁMETROS DE LA NEURONA
# ---------------------------------------------------------------------
# Una neurona (perceptrón) tiene:
#   - Un peso (w) por cada entrada: mide qué tan importante es esa
#     entrada para la decisión.
#   - Un sesgo o bias (b): permite mover la frontera de decisión,
#     independientemente de las entradas (equivale al "intercepto"
#     en una recta y = mx + b).
#
# Al inicio, la neurona NO sabe nada: sus pesos son valores
# pequeños y aleatorios. El entrenamiento es el proceso de ir
# ajustando estos números hasta que la neurona clasifique bien.
random.seed(42)  # para que el resultado sea reproducible en clase

w1 = random.uniform(-1, 1)  # peso asociado a x1 (horas de estudio)
w2 = random.uniform(-1, 1)  # peso asociado a x2 (horas de práctica)
b = random.uniform(-1, 1)   # bias (sesgo)

print(f"Pesos iniciales (aleatorios): w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")


# ---------------------------------------------------------------------
# 3) LA MATEMÁTICA DE LA NEURONA: SUMA PONDERADA + ACTIVACIÓN
# ---------------------------------------------------------------------
def suma_ponderada(x1, x2, w1, w2, b):
    """
    Esto es EXACTAMENTE lo que hace una neurona antes de "decidir":
    multiplica cada entrada por su peso, suma todo, y le suma el bias.

        z = (x1 * w1) + (x2 * w2) + b

    z es un solo número. Si z es grande y positivo, la neurona
    "tiende" hacia la clase 1. Si es muy negativo, tiende hacia
    la clase 0. Pero z todavía no es la decisión final: falta
    pasarlo por la función de activación.
    """
    z = (x1 * w1) + (x2 * w2) + b
    return z


def funcion_activacion_escalon(z):
    """
    La función de activación convierte el número z en una decisión
    binaria concreta: 0 o 1.

    Usamos la función escalón (step function), la más simple posible
    y la que históricamente usó el perceptrón original de Rosenblatt:

        activación(z) = 1  si z >= 0
        activación(z) = 0  si z <  0

    Nota para la clase: esta función NO es diferenciable en z=0,
    por eso las redes modernas usan otras activaciones (sigmoid,
    ReLU, etc.) que sí se pueden usar con descenso de gradiente.
    Aquí usamos el escalón porque es la más fácil de visualizar
    a mano, y el perceptrón clásico no necesita gradientes para
    aprender (ver la regla de aprendizaje más abajo).
    """
    return 1 if z >= 0 else 0


def predecir(x1, x2, w1, w2, b):
    """Une los dos pasos anteriores: suma ponderada -> activación."""
    z = suma_ponderada(x1, x2, w1, w2, b)
    return funcion_activacion_escalon(z)


# ---------------------------------------------------------------------
# 4) EL ENTRENAMIENTO: LA REGLA DE APRENDIZAJE DEL PERCEPTRÓN
# ---------------------------------------------------------------------
# Idea central: en cada ejemplo del dataset, comparamos lo que la
# neurona PREDIJO contra lo que REALMENTE debía responder (el error).
#
#     error = y_real - y_predicho
#
# Si error = 0 -> la neurona acertó, no se toca nada.
# Si error = 1 -> la neurona dijo 0 y debía decir 1 (le faltó "empuje").
# Si error = -1 -> la neurona dijo 1 y debía decir 0 (le sobró "empuje").
#
# Ajustamos cada peso en proporción al error y al valor de la entrada
# que lo acompaña, usando una tasa de aprendizaje (learning rate):
#
#     w1_nuevo = w1 + tasa_aprendizaje * error * x1
#     w2_nuevo = w2 + tasa_aprendizaje * error * x2
#     b_nuevo  = b  + tasa_aprendizaje * error * 1     (el bias no tiene x)
#
# Esta es la versión más simple posible de "aprender de los datos":
# no hay derivadas ni backpropagation todavía (eso viene en la
# Sesión 2 con redes de más de una neurona), pero el espíritu es
# el mismo: usar el error para corregir los parámetros.

tasa_aprendizaje = 0.1
numero_de_epocas = 10  # una "época" = pasar una vez por todo el dataset

print("\n--- ENTRENAMIENTO ---")
for epoca in range(numero_de_epocas):
    errores_en_esta_epoca = 0

    for x1, x2, y_real in datos_entrenamiento:
        y_predicho = predecir(x1, x2, w1, w2, b)
        error = y_real - y_predicho

        if error != 0:
            errores_en_esta_epoca += 1
            # Ajuste de cada parámetro según la regla explicada arriba
            w1 = w1 + tasa_aprendizaje * error * x1
            w2 = w2 + tasa_aprendizaje * error * x2
            b = b + tasa_aprendizaje * error * 1

    print(f"Época {epoca + 1}: errores = {errores_en_esta_epoca}, "
          f"w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")

    # Si en una época completa no hubo ningún error, la neurona ya
    # encontró una línea que separa perfectamente las dos clases:
    # podemos parar antes de tiempo.
    if errores_en_esta_epoca == 0:
        print("La neurona ya no comete errores, entrenamiento terminado.")
        break


# ---------------------------------------------------------------------
# 5) PROBAR LA NEURONA YA ENTRENADA CON DATOS NUEVOS
# ---------------------------------------------------------------------
print("\n--- PRUEBA CON ESTUDIANTES NUEVOS (no vistos en el entrenamiento) ---")
estudiantes_nuevos = [
    (1, 2),  # pocas horas de estudio y práctica -> se espera 0
    (5, 4),  # muchas horas -> se espera 1
    (3, 1),  # caso intermedio, interesante para discutir en clase
]

for x1, x2 in estudiantes_nuevos:
    z = suma_ponderada(x1, x2, w1, w2, b)
    resultado = funcion_activacion_escalon(z)
    etiqueta = "APRUEBA" if resultado == 1 else "NO APRUEBA"
    print(f"Estudiante(horas_estudio={x1}, horas_practica={x2}) "
          f"-> z={z:.3f} -> {etiqueta}")
