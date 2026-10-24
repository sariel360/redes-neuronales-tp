# Clasificador de imágenes: placas Arduino vs. Raspberry Pi

**Proyecto Personal 1: Redes neuronales con Python y TensorFlow**

| | |
|---|---|
| **Institución** | IPET N°249 "Nicolás Copérnico", Córdoba, Argentina |
| **Plataforma** | Google Colab |
| **Lenguaje y librerías** | Python 3, TensorFlow / Keras (2.21.0), scikit-learn, Matplotlib, Seaborn |
| **Cuaderno** | [https://colab.research.google.com/drive/1mkXeiU_sqZGyrpXWUPx-3S-tznHZw45G?usp=sharing] |
| **Dataset** | [Carpeta de Google Drive](https://drive.google.com/drive/folders/101YpfbQtVdEwuYelNpaz8Cr0fdDNPvTd?hl=es) |

---

## Índice

1. [Resumen](#1-resumen)
2. [Objetivo](#2-objetivo)
3. [Marco teórico](#3-marco-teórico)
4. [Dataset](#4-dataset)
5. [Metodología](#5-metodología)
6. [Resultados](#6-resultados)
7. [Análisis de resultados](#7-análisis-de-resultados)
8. [Limitaciones y mejoras posibles](#8-limitaciones-y-mejoras-posibles)
9. [Conclusiones](#9-conclusiones)
10. [Cómo reproducir el proyecto](#10-cómo-reproducir-el-proyecto)
11. [Declaración de uso de herramientas de IA](#11-declaración-de-uso-de-herramientas-de-ia)

---

## 1. Resumen

Este proyecto implementa un **modelo de clasificación de imágenes** basado en una red neuronal convolucional (CNN) que distingue entre dos tipos de placas electrónicas: **Arduino** y **Raspberry Pi**. Se armó un dataset propio de **339 imágenes** (150 de Arduino y 189 de Raspberry Pi), se entrenó el modelo en Google Colab y se evaluó con 67 imágenes que el modelo nunca vio durante el entrenamiento.

El modelo alcanzó una **exactitud del 83,58 %** en validación. Los errores se concentraron en fotos de primer plano, imágenes con otros componentes alrededor de la placa, kits e ilustraciones poco representativas, lo que muestra la importancia de la calidad del dataset.

## 2. Objetivo

Implementar un clasificador de imágenes aplicando los conceptos vistos en el material audiovisual de la clase, creando, procesando y analizando un conjunto de datos propio. Los objetivos específicos fueron:

- Seleccionar una temática de interés y recopilar imágenes en al menos dos categorías.
- Entrenar una red neuronal en Google Colab con el dataset guardado en Google Drive.
- Evaluar el rendimiento del modelo e interpretar sus aciertos y errores.

**Temática elegida:** electrónica y programación de placas de desarrollo. Las dos categorías son visualmente parecidas (ambas son circuitos impresos con chips y pines), lo que hace que el problema sea más interesante que distinguir objetos muy diferentes entre sí.

## 3. Marco teórico

**Red neuronal convolucional (CNN).** Es un tipo de red neuronal diseñada para trabajar con imágenes. Una imagen es una grilla de números (los píxeles), y las capas convolucionales aplican filtros que detectan patrones simples (bordes, colores, texturas). Las capas siguientes combinan esos patrones en formas más complejas, hasta que la capa final calcula la probabilidad de que la imagen pertenezca a cada categoría.

**Conceptos usados en el proyecto:**

| Concepto | Para qué sirve |
|---|---|
| `Conv2D` | Detecta patrones visuales mediante filtros |
| `MaxPooling2D` | Reduce el tamaño de la imagen conservando lo importante |
| `Dropout` | Apaga neuronas al azar para reducir el sobreajuste |
| `softmax` | Convierte la salida en probabilidades que suman 100 % |
| Data augmentation | Genera variaciones de las imágenes (giros, zoom) para que el modelo generalice mejor |
| Validación | Reserva imágenes que el modelo no usa para aprender, para medir su rendimiento real |
| `EarlyStopping` | Detiene el entrenamiento cuando la validación deja de mejorar |
| Sobreajuste (*overfitting*) | Ocurre cuando el modelo memoriza el entrenamiento y falla con imágenes nuevas |
| Matriz de confusión | Tabla que cruza la clase real con la predicha para ver qué errores comete el modelo |
| Precisión, recall y F1 | Métricas por categoría: qué tan confiables son las predicciones, qué porcentaje de cada clase se detecta y el equilibrio entre ambas |

## 4. Dataset

### 4.1 Origen y preparación

Las imágenes se descargaron manualmente desde buscadores de internet usando la extensión de Chrome **Download All Images**, guardando cada categoría en su propia carpeta. Luego se hizo una **depuración con ayuda de ChatGPT** para descartar las imágenes que no correspondían a la categoría (por ejemplo, fotos de otros objetos); el detalle de cómo se usó está en la [sección 11](#11-declaración-de-uso-de-herramientas-de-ia). El resultado se guardó en Google Drive con la siguiente estructura:

```
MyDrive/
└── dataset_proyecto/
    ├── arduino/          150 imágenes
    └── raspberry_pi/     189 imágenes
```

El nombre de cada subcarpeta funciona como **etiqueta** de la categoría: Keras lo lee automáticamente.

### 4.2 Composición

| Categoría | Imágenes | Porcentaje |
|---|---|---|
| `arduino` | 150 | 44 % |
| `raspberry_pi` | 189 | 56 % |
| **Total** | **339** | 100 % |

El dataset está **levemente desbalanceado**: Raspberry Pi tiene un 26 % más de imágenes que Arduino. Se mantuvo así y se analizó su efecto en los resultados.

### 4.3 División entrenamiento / validación

| Conjunto | Imágenes |
|---|---|
| Entrenamiento (80 %) | 272 |
| Validación (20 %) | 67 |

La división se hizo con una semilla fija (`seed=42`) para que sea reproducible. En la validación quedaron 26 imágenes de Arduino y 41 de Raspberry Pi.

## 5. Metodología

### 5.1 Parámetros

| Parámetro | Valor |
|---|---|
| Tamaño de imagen | 128 × 128 píxeles (RGB) |
| Tamaño de lote (*batch*) | 32 |
| Épocas máximas | 30 |
| Optimizador | Adam |
| Función de pérdida | `sparse_categorical_crossentropy` |
| Parada temprana | `EarlyStopping` sobre `val_loss`, paciencia 5, restaurando los mejores pesos |

### 5.2 Arquitectura del modelo

```
Entrada (128 × 128 × 3)
 ├─ Data augmentation: RandomFlip horizontal, RandomRotation 0.1, RandomZoom 0.1
 ├─ Rescaling 1/255
 ├─ Conv2D (32 filtros, 3×3, ReLU)  → MaxPooling2D
 ├─ Conv2D (64 filtros, 3×3, ReLU)  → MaxPooling2D
 ├─ Conv2D (128 filtros, 3×3, ReLU) → MaxPooling2D
 ├─ Dropout 0.3
 ├─ Flatten
 ├─ Dense 128 (ReLU)
 └─ Dense 2 (softmax)  → [P(arduino), P(raspberry_pi)]
```

El modelo tiene **3.304.898 parámetros entrenables** (12,61 MB). Casi todos (≈ 3,2 millones) están en la capa `Dense 128`, que recibe la imagen aplanada.

### 5.3 Flujo de trabajo en el cuaderno

1. Importación de librerías y montaje de Google Drive.
2. Verificación del dataset (cantidad de imágenes por categoría).
3. Carga con `image_dataset_from_directory` y visualización de ejemplos.
4. Construcción y compilación del modelo.
5. Entrenamiento con `EarlyStopping`.
6. Curvas de aprendizaje.
7. Evaluación: exactitud, matriz de confusión, reporte por categoría e imágenes mal clasificadas.
8. Prueba con una imagen nueva subida manualmente.
9. Informe de análisis en una celda de texto (Markdown) al final del cuaderno.

## 6. Resultados

### 6.1 Resultado general

| Métrica | Valor |
|---|---|
| Imágenes de validación | 67 |
| Aciertos | 56 |
| Errores | 11 |
| **Exactitud (accuracy)** | **83,58 %** |
| Margen de error | 16,42 % |
| Época de los pesos evaluados | 12 (la de menor `val_loss`: 0,376) |
| Precisión de entrenamiento en esa época | ≈ 81 % (≈ 89 % en la época 17, la última que se ejecutó) |

> **Cómo leer estos números.** El entrenamiento se detuvo en la época 17, pero `EarlyStopping` restauró los pesos de la época 12, que fue la mejor en validación (83,58 %, el mismo valor que se midió en la evaluación final). Por eso la exactitud final no coincide con la última época, donde la validación había caído a 67 %.

Como referencia, un modelo "tramposo" que contestara siempre `raspberry_pi` acertaría 41 de 67 imágenes (**61,2 %**). El 83,58 % lo supera por más de 20 puntos, así que la red aprendió algo real y no solo la proporción de las clases.

### 6.2 Matriz de confusión

| | Predicho: `arduino` | Predicho: `raspberry_pi` |
|---|:---:|:---:|
| **Real: `arduino`** | 19 | 7 |
| **Real: `raspberry_pi`** | 4 | 37 |

![Matriz de confusión](img/matriz.png)

### 6.3 Métricas por categoría

| Categoría | Precisión | Recall | F1-score | Imágenes |
|---|:---:|:---:|:---:|:---:|
| `arduino` | 0,83 | 0,73 | 0,78 | 26 |
| `raspberry_pi` | 0,84 | 0,90 | 0,87 | 41 |
| **Promedio macro** | 0,83 | 0,82 | 0,82 | 67 |

### 6.4 Curvas de aprendizaje

La precisión de entrenamiento subió de forma sostenida (de ≈ 51 % a ≈ 89 %) y la pérdida bajó de 0,97 a 0,27. La curva de **validación fue muy inestable**: arrancó en 39 %, después osciló entre ≈ 55 % y ≈ 84 % y en la última época cayó a 67 %. Esto es típico cuando el conjunto de validación es chico (67 imágenes): cada pequeño cambio en los pesos mueve varias predicciones a la vez. El entrenamiento se detuvo por `EarlyStopping` en la época 17.

![Curvas de precisión y pérdida](img/curvas.png)

### 6.5 Imágenes mal clasificadas

![Imágenes mal clasificadas](img/errores.png)

| # | Real | Predicho (confianza) | Qué muestra la imagen | Causa probable |
|:-:|---|---|---|---|
| 1 | `raspberry_pi` | `arduino` (73 %) | Afiche "Dead?" con una Raspberry Pi y efectos gráficos | Ilustración con texto, poco representativa |
| 2 | `raspberry_pi` | `arduino` (88 %) | La misma imagen del afiche | **Duplicado** en el dataset |
| 3 | `arduino` | `raspberry_pi` (54 %) | Miniatura de un tutorial "Motion Sensor Relay" con sensor y módulo relé | Texto y accesorios alrededor de la placa |
| 4 | `arduino` | `raspberry_pi` (70 %) | Caja organizadora con un kit de componentes | Kit: la placa casi no se distingue |
| 5 | `arduino` | `raspberry_pi` (67 %) | Primer plano de los pines (`POWER`, `ANALOG IN`) | Primer plano: no se ve la forma de la placa |
| 6 | `arduino` | `raspberry_pi` (60 %) | Placa con joystick, sensores y cables sueltos | Accesorios que compiten con la placa |
| 7 | `raspberry_pi` | `arduino` (73 %) | Primer plano de placas con sus puertos | Primer plano y encuadre parcial |
| 8 | `arduino` | `raspberry_pi` (72 %) | Persona con una placa chica frente a un monitor, fondo borroso | La placa es un detalle pequeño de una escena |
| 9 | `arduino` | `raspberry_pi` (63 %) | Varias placas pequeñas juntas | Placas amontonadas; modelos chicos parecidos |
| 10 | `raspberry_pi` | `arduino` (68 %) | Raspberry Pi con pantalla OLED y cable USB | El accesorio llama más la atención que la placa |
| 11 | `arduino` | `raspberry_pi` (54 %) | Arduino rodeado de protoboard, servo, pantalla LCD y sensores | Kit con muchos componentes |

La **confianza promedio en los errores fue de ≈ 67 %** y solo uno superó el 80 % (justamente el duplicado). Es decir, en la mayoría de los casos el modelo dudaba.

### 6.6 Prueba con una imagen nueva

Se subió una foto de una placa Arduino Uno que **no pertenecía al dataset**. El modelo la clasificó como `arduino` con 100 % de confianza.

## 7. Análisis de resultados

### ¿El modelo logró clasificar correctamente las imágenes?

En general sí: acertó 56 de 67 imágenes de validación (83,58 %), muy por encima del 61,2 % que lograría un modelo que siempre responda la clase más frecuente. Reconoce mejor las Raspberry Pi (recall 0,90) que los Arduino (recall 0,73).

### ¿Qué margen de error se notó?

El margen de error fue de **16,42 %** (11 imágenes). Hubo **7 Arduino clasificados como Raspberry Pi** y **4 Raspberry Pi clasificadas como Arduino**. Además, la validación fue inestable entre épocas (entre ≈ 55 % y ≈ 84 %), por lo que conviene tomar el 83,58 % como una estimación aproximada y no como un valor exacto.

### ¿A qué se deben los falsos positivos y falsos negativos?

Al revisar las 11 imágenes mal clasificadas (tabla de la sección 6.5) se identificaron estas causas:

- **Placa que no es el protagonista de la imagen:** en 8 de los 11 errores la placa aparece rodeada de otros elementos (kits, sensores, cables, pantallas, una persona, texto, efectos gráficos). El modelo se fija en los accesorios y no en la placa.
- **Fotos de primer plano:** en tomas muy cercanas de pines o puertos se pierde la forma completa y quedan texturas que ambas placas comparten (conectores, circuitos, soldaduras).
- **Desbalance entre categorías:** con 189 imágenes de Raspberry Pi contra 150 de Arduino, el modelo tiende a inclinarse hacia la clase con más ejemplos. Coincide con que la mayoría de los errores (7 de 11) fueron Arduino clasificados como Raspberry Pi.
- **Imágenes poco representativas y duplicadas:** ilustraciones con texto (como el afiche "Dead?") se parecen poco al resto del set, y esa misma imagen aparece dos veces entre los errores, señal de que la limpieza dejó duplicados.
- **Placas amontonadas y rasgos compartidos:** una foto con varias placas pequeñas juntas confundió al modelo, y hay modelos de tamaño parecido (Arduino Nano y Raspberry Pi Pico). Además, ambas son circuitos con pines y chips.
- **Depuración parcial del dataset:** la limpieza de imágenes fue asistida y sin revisión exhaustiva, por lo que quedaron imágenes poco útiles para el entrenamiento.
- **Dataset acotado y resolución baja:** con unas 340 imágenes de calidad dispar, redimensionadas a 128×128 píxeles, se pierden detalles finos que distinguen una placa de otra.

## 8. Limitaciones y mejoras posibles

**Limitaciones**
- Un conjunto de validación de solo 67 imágenes hace que cada error mueva el resultado unos 1,5 puntos porcentuales.
- El resultado corresponde a una única ejecución: el sorteo aleatorio y el entrenamiento pueden dar variaciones.
- Si hay imágenes duplicadas en el dataset, pueden haber quedado copias en entrenamiento y en validación a la vez. En ese caso el modelo "vería" en validación una imagen que ya estudió y el 83,58 % saldría algo inflado.
- El modelo solo conoce dos categorías: ante una imagen de otra cosa, igual elegirá una de las dos.

**Mejoras propuestas**
1. **Balancear** las categorías (misma cantidad de imágenes en ambas).
2. **Ampliar** el dataset con más imágenes limpias y fotos propias tomadas con celular.
3. **Eliminar** duplicados, ilustraciones, kits y fotos donde la placa no sea claramente visible.
4. Aumentar la **resolución** de entrada (por ejemplo, 160×160).
5. Usar **transfer learning** con un modelo preentrenado (por ejemplo, MobileNetV2), que rinde mejor con pocos datos.
6. Aplicar **validación cruzada** para obtener una medición más estable.
7. Marcar como "dudosa" toda predicción con confianza baja (por ejemplo, menos de 75 %) en lugar de aceptarla sin más: casi todos los errores de este modelo estuvieron en ese rango.

## 9. Conclusiones

El proyecto permitió recorrer el ciclo completo de un clasificador de imágenes: armar un dataset, entrenar una CNN, evaluarla e interpretar sus errores. El modelo aprendió a diferenciar placas Arduino y Raspberry Pi en situaciones comunes con un 83,58 % de exactitud, pero falla cuando la foto no muestra la placa completa o aparece rodeada de otros elementos.

La principal enseñanza es que **la calidad y el balance del dataset pesan tanto como la arquitectura del modelo**: gran parte de los errores se explican por imágenes poco representativas, duplicadas y por la diferencia de cantidad entre categorías, más que por la red en sí.

## 10. Cómo reproducir el proyecto

1. Crear en Google Drive la carpeta `dataset_proyecto` con una subcarpeta por categoría (`arduino`, `raspberry_pi`) y cargar las imágenes.
2. Abrir el cuaderno en Google Colab.
3. Verificar que `DATASET_PATH` coincida con la ubicación de la carpeta en Drive (si la carpeta no está en la raíz de "Mi unidad", hay que agregar la subcarpeta a la ruta):
   ```python
   DATASET_PATH = '/content/drive/MyDrive/dataset_proyecto'
   ```
4. Ejecutar todas las celdas (*Entorno de ejecución → Ejecutar todas*) y autorizar el acceso a Drive cuando lo solicite.
5. Para probar una imagen nueva, usar la celda de predicción y subir una foto desde la computadora.

> Nota: los resultados pueden variar levemente entre ejecuciones por el componente aleatorio del entrenamiento.

## 11. Declaración de uso de herramientas de IA

### 11.1 Resumen

| Herramienta | Dónde se usó | Para qué se usó |
|---|---|---|
| ChatGPT | Chat externo | Ayuda para revisar y depurar las imágenes del dataset |
| Gemini | Dentro de Google Colab ("Explica este error") | Entender un `FileNotFoundError` al cargar el dataset desde Drive |
| Claude (Anthropic) | Chat externo | Estructurar y comentar el código del cuaderno, interpretar resultados y redactar este informe |

### 11.2 Detalle

**ChatGPT: selección y depuración de imágenes**

- Se le subieron las páginas guardadas de los resultados de búsqueda (Brave Search) de "arduino" y de "raspberry pi". ChatGPT analizó los archivos y armó grillas numeradas con miniaturas para poder revisar las imágenes y aplicar criterios de selección.
- Luego se le pidió eliminar las imágenes duplicadas o innecesarias de la carpeta de Drive. ChatGPT aclaró que **no podía acceder a Google Drive ni modificar los archivos**, y propuso criterios de limpieza: eliminar copias exactas, conservar una sola de las imágenes casi idénticas, descartar imágenes borrosas, corruptas o que no muestren el tema, y conservar las variaciones útiles de ángulo e iluminación.

| Selección de imágenes de Arduino | Selección de imágenes de Raspberry Pi | Pedido de limpieza de la carpeta de Drive |
|:-:|:-:|:-:|
| ![ChatGPT, imágenes de Arduino](img/ia/chatgpt-seleccion-arduino.jpg) | ![ChatGPT, imágenes de Raspberry Pi](img/ia/chatgpt-seleccion-raspberry-pi.jpg) | ![ChatGPT, limpieza de Drive](img/ia/chatgpt-limpieza-drive.jpg) |

Como la depuración fue asistida y no exhaustiva, el dataset final todavía contiene duplicados e imágenes poco útiles (ver secciones 6.5 y 7).

**Gemini (en Colab): error de ruta**

Al montar Drive y ejecutar la verificación del dataset apareció `FileNotFoundError: [Errno 2] No such file or directory` para la ruta `/content/drive/MyDrive/dataset_proyecto`. Con la función "Explica este error" de Colab, Gemini indicó tres causas posibles: el nombre de la carpeta no coincide exactamente (mayúsculas, minúsculas, espacios), la carpeta está dentro de otra subcarpeta de "Mi unidad" y hay que agregarla a la ruta, o Drive todavía no sincronizó los archivos. Una vez corregida la ruta, la celda funcionó (150 imágenes de Arduino y 189 de Raspberry Pi).

![Gemini, explicación del error](img/ia/gemini-error-ruta.png)

**Claude (Anthropic)**

- Asistencia para estructurar y comentar el código del cuaderno, interpretar resultados y redactar este informe.
- En la conversación más reciente se le entregaron el enunciado del trabajo, el cuaderno de Colab, las capturas de resultados (curvas, matriz de confusión e imágenes mal clasificadas), el borrador de este README y las capturas de las conversaciones con ChatGPT y Gemini. Claude revisó que las cifras del informe coincidieran con las salidas del cuaderno (por ejemplo, que el 83,58 % corresponde a los pesos restaurados de la época 12 y no a la última época), corrigió el rango de oscilación de la validación, agregó la tabla de errores con su análisis, redactó la celda final de análisis del cuaderno y escribió esta declaración.

### 11.3 Qué hizo el autor

Las imágenes fueron seleccionadas y descargadas por el autor. Las decisiones sobre la temática, las categorías y la interpretación final de los resultados son del autor. El entrenamiento, la evaluación y las capturas de resultados provienen de la ejecución propia del cuaderno en Google Colab.

---

*Proyecto realizado en el marco de la materia correspondiente al Proyecto Personal 1 — IPET N°249 "Nicolás Copérnico".*
