# Deep Learning — Fundamentos y Ejemplos

Código y ejemplos del módulo de **Deep Learning**. Pensado como material de clase y como referencia para revisar después de cada sesión.

## Contenido del repositorio

| Carpeta / archivo | Descripción |
|---|---|
| `01_perceptron/` | Perceptrón simple en Python puro, sin frameworks
| `02_entrenamiento/` | Optimización y regularización (Keras/TensorFlow)
| `03_cnn/` | Redes convolucionales y transfer learning
| `04_atencion/` | De RNN a mecanismos de atención
| `05_llm_generative/` | LLMs
| `06_proyecto_final/` | Ejemplos de referencia para el proyecto final

> Cada carpeta incluye su propio notebook o script comentado.

## Requisitos

- Python 3.10+
- [Google Colab](https://colab.research.google.com/) (recomendado, no requiere instalación local) o entorno local con:

```bash
pip install -r requirements.txt
```

Librerías principales usadas en el curso: `tensorflow`, `keras`, `numpy`, `matplotlib`, `scikit-learn`.

## Cómo usar este repositorio

1. Cloná el repositorio:
   ```bash
   git clone https://github.com/<tu-usuario>/<nombre-repo>.git
   ```
2. Entrá a la carpeta de la sesión que corresponda.
3. Abrí el notebook en Colab o Jupyter, o corré el script directamente:
   ```bash
   python nombre_del_ejemplo.py
   ```
4. Los comentarios en el código explican tanto el *qué* como el *por qué* de cada paso — están pensados para leerse, no solo para ejecutarse.

## Estructura sugerida por ejemplo

Cada ejemplo nuevo que se agregue debería mantener este formato:

- Comentario inicial explicando el objetivo del ejemplo y qué concepto ilustra.
- Sección de datos de entrada, comentada.
- Sección de modelo/arquitectura, comentada.
- Sección de entrenamiento, con métricas visibles por época.
- Sección final de prueba/inferencia con datos nuevos.
- Bloque de "puntos para discutir en clase" al cierre del archivo.

## Recursos de referencia

- Chollet, François. *Deep Learning with Python*, Manning, 2021.
- Géron, Aurélien. *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*, O'Reilly, 2023.
- Fleuret, François. *The Little Book of Deep Learning*, Université de Genève, 2023.
- Tunstall, von Werra, Wolf. *Natural Language Processing with Transformers*, O'Reilly, 2022.
- [Keras — documentación oficial](https://keras.io/)
- [Hugging Face](https://huggingface.co/)

## Autora

MsC. Ing. Lesly Zerna
Diplomado Deep Learning

## Licencia

Material educativo de uso libre para fines académicos. Se agradece atribución si se reutiliza fuera del diplomado.
