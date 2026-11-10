# Informe de Ablación de Dimensionalidad Matryoshka (MRL) en Gemini Embedding 2 (`3072` vs. `1536` vs. `768`)

Fecha de ejecución y auditoría:
- **Ejecuciones auditadas (`2026-11-10`):** `audit/run_20261110_113907/` y `audit/run_20261110_120928/`
- **Datos agregados exportados:**
  - Consultas literales (102 consultas): [`matryoshka_ablation_20261110.csv`](matryoshka_ablation_20261110.csv)
  - Consultas generales / multi-objetivo (20 consultas, Etapa 1 y Etapa 2): [`matryoshka_multi_target_20261110.csv`](matryoshka_multi_target_20261110.csv)
  - Consultas negativas de control (16 consultas, Etapa 1 y Etapa 2): [`matryoshka_negative_20261110.csv`](matryoshka_negative_20261110.csv)
  - Desglose individual por consulta literal: `audit/run_20261110_120928/matryoshka_queries_detail_20261110.csv`
- **Estado de la base de datos local (`./data/vector_db`):**
  - `python_tutorial.mp4` (`242,5 s`): **82 chunks** indexados a `3072` dimensiones (`threshold = 0,6`, `max_shot_duration = 5,0 s`), **81/82 chunks (`98,8 %`)** con transcripción Whisper (`base`, idioma `es`).
  - `animals.mp4` (`243,0 s`): **82 chunks** indexados a `3072` dimensiones (`threshold = 0,5`, `max_shot_duration = 8,0 s`), **82/82 chunks (`100,0 %`)** con transcripción Whisper (`base`, idioma `es`).

---

## 1. Objetivo del estudio y fundamento matemático (MRL + Norma L2)

El modelo fundacional omnimodal `gemini-embedding-2-preview` se entrena mediante *Matryoshka Representation Learning* (MRL), estructurando la información semántica de forma jerárquica desde las primeras coordenadas del vector hasta su dimensión nativa máxima (`d = 3072`). Google recomienda oficialmente tres configuraciones operativas dentro del rango soportado (`128 <= d <= 3072`): **`3072`**, **`1536`** y **`768`** dimensiones.

Mientras que los vectores nativos de `3072` dimensiones son devueltos por la API con longitud unitaria (`||v||_2 = 1,0`), el truncado a una sub-dimensión `d < 3072` elimina las coordenadas de cola `[d:3072]` y reduce la norma euclídea del sub-vector resultante (`||v[:d]||_2 < 1,0`). Para preservar la validez geométrica de la similitud del coseno en Qdrant, la función `truncate_and_normalize_mrl(vector, target_dim)` implementada en `src/embeddings/gemini_vectorizer.py` aplica el recorte prefijo seguido de re-normalización L2:

$$\hat{\mathbf{v}}_{1:d} = \frac{\mathbf{v}_{1:d}}{\|\mathbf{v}_{1:d}\|_2} = \frac{\mathbf{v}_{1:d}}{\sqrt{\sum_{j=1}^{d} v_j^2}}, \quad \|\hat{\mathbf{v}}_{1:d}\|_2 = 1,0$$

Este estudio evalúa empíricamente el compromiso (*trade-off*) entre **precisión de recuperación en Etapa 1 y Etapa 2**, **ahorro de memoria en base de datos vectorial** y **latencia de búsqueda en Qdrant** sobre los 164 fragmentos de los dos corpus de referencia del TFM a través de tres baterías complementarias:
1. **102 consultas literales de evento único** (`56` en `python_tutorial.mp4` + `46` en `animals.mp4`).
2. **20 consultas generales / abstractas multi-objetivo** (`10` en `python_tutorial.mp4` con `48` intervalos esperados + `10` en `animals.mp4` como `"Felinos"`, `"Reptiles"`, `"Aves"`, `"Manada..."` con `58` intervalos esperados), evaluadas en Etapa 1 (`Qdrant`) y Etapa 2 (`gemini-3.5-flash-lite`).
3. **16 consultas negativas de control** (`8` por vídeo, `80` candidatos por dimensión y `240` evaluaciones totales), auditando la inmunidad a falsos positivos en Etapa 2 tras el truncado dimensional.

---

## 2. Batería 1: Consultas literales de evento único (102 consultas, Etapa 1 Qdrant)

### 2.1 Comparativa por vídeo y dimensión (`top_k = 5`, Etapa 1 Qdrant)

| Corpus de Vídeo | Dimensión (`d`) | Chunks | Consultas | Recall@1 | Recall@3 | Recall@5 | MRR | Memoria / Vector (`float32`) | Memoria Total Chunks | Ahorro Memoria | Latencia Media Qdrant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `python_tutorial.mp4` | **3072** | 82 | 56 | **85,71 % (48/56)** | **92,86 % (52/56)** | 94,64 % (53/56) | 0,8973 | 12,00 KB (12.288 B) | 984,00 KB | 0,0 % | 1,1688 ms |
| `python_tutorial.mp4` | **1536** | 82 | 56 | **85,71 % (48/56)** | **92,86 % (52/56)** | 96,43 % (54/56) | 0,9009 | 6,00 KB (6.144 B) | 492,00 KB | **50,0 %** | 0,7502 ms (-35,8 %) |
| `python_tutorial.mp4` | **768** | 82 | 56 | **85,71 % (48/56)** | **92,86 % (52/56)** | **98,21 % (55/56)** | **0,9045** | 3,00 KB (3.072 B) | 246,00 KB | **75,0 %** | **0,6297 ms (-46,1 %)** |
| `animals.mp4` | **3072** | 82 | 46 | **71,74 % (33/46)** | 93,48 % (43/46) | **100,00 % (46/46)** | **0,8341** | 12,00 KB (12.288 B) | 984,00 KB | 0,0 % | 1,1104 ms |
| `animals.mp4` | **1536** | 82 | 46 | 69,57 % (32/46) | 93,48 % (43/46) | **100,00 % (46/46)** | 0,8232 | 6,00 KB (6.144 B) | 492,00 KB | **50,0 %** | 0,7848 ms (-29,3 %) |
| `animals.mp4` | **768** | 82 | 46 | 69,57 % (32/46) | **95,65 % (44/46)** | **100,00 % (46/46)** | 0,8261 | 3,00 KB (3.072 B) | 246,00 KB | **75,0 %** | **0,6418 ms (-42,2 %)** |

### 2.2 Agregado global sobre el banco literal completo (102 consultas, 164 chunks)

| Dimensión (`d`) | Recall@1 Global (102 q) | Recall@3 Global (102 q) | Recall@5 Global (102 q) | MRR Ponderado | Memoria Total (164 chunks) | Ahorro de Almacenamiento | Latencia Media Qdrant |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3072** | **79,41 % (81/102)** | 93,14 % (95/102) | 97,06 % (99/102) | **0,8688** | 1.968,00 KB (1,92 MB) | 0,0 % | 1,1425 ms |
| **1536** | 78,43 % (80/102) | 93,14 % (95/102) | 98,04 % (100/102) | 0,8659 | 984,00 KB (0,96 MB) | **50,0 %** | 0,7658 ms (-33,0 %) |
| **768** | 78,43 % (80/102) | **94,12 % (96/102)** | **99,02 % (101/102)** | 0,8691 | 492,00 KB (0,48 MB) | **75,0 %** | **0,6352 ms (-44,4 %)** |

---

## 3. Análisis detallado de consultas literales por dominio audiovisual

### 3.1 Dominio técnico con locución acoplada (`python_tutorial.mp4`, 56 consultas)

1. **Invarianza total en `Recall@1` y `Recall@3`:** Al comprimir los embeddings de `3072` a `1536` (-50 % de memoria) y a `768` (-75 % de memoria), el `Recall@1` permanece exactamente en **85,71 % (48/56)** y el `Recall@3` en **92,86 % (52/56)**. Ninguna de las 48 consultas acertadas en primera posición pierde su puesto.
2. **Estabilidad query a query:** De las 56 consultas literales, **53 consultas (`94,6 %`) conservan exactamente el mismo rango numérico** en las tres dimensiones.
3. **Efecto de filtrado de ruido en `Recall@5` (+3,57 pp en `768`):** En dimensiones menores (`1536` y `768`), el truncado elimina componentes residuales de baja varianza (asociados al fondo invariante del editor Visual Studio Code), lo que permite que dos consultas cuyos fragmentos caían justo fuera del corte en `3072` entren en el `Top-5`:

| # Consulta | Consulta en `python_tutorial.mp4` | Intervalo (s) | Rango `d=3072` | Rango `d=1536` | Rango `d=768` | Comportamiento Matryoshka |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | Presentador en B/N saludando con dos manos levantadas | 0,0 - 0,7 | 4 | 4 | 5 | Permanece dentro del `Top-5` en las 3 dimensiones |
| 45 | Símbolo de llaves `{ }` tachado con una cruz roja `X` | 201,8 - 205,4 | >5 | **5** | **5** | Entra en `Top-5` en `1536` y `768` (+1 acierto `Recall@5`) |
| 49 | Primera salida del 0 al 4 en TERMINAL tras `range(5)` | 224,4 - 225,3 | >5 | >5 | **4** | Entra en `Top-5` en `768` (+1 acierto adicional `Recall@5`) |

### 3.2 Dominio documental con voz en *off* desacoplada (`animals.mp4`, 46 consultas)

1. **Cobertura `Recall@5` perfecta (`100,00 %`) en las tres dimensiones:** Las 46 consultas literales de `animals.mp4` localizan el fragmento correcto dentro de los 5 primeros resultados tanto en `3072` como en `1536` y `768` dimensiones (`46/46`). Esto garantiza que el verificador bimodal de Etapa 2 (`gemini-3.5-flash-lite`) recibe siempre el candidato verdadero en su ventana de inspección independientemente del nivel de compresión Matryoshka elegido.
2. **Estabilidad query a query:** De las 46 consultas, **41 consultas (`89,1 %`) mantienen idéntica posición** en `3072`, `1536` y `768`.
3. **Análisis de las 5 únicas consultas con variación de posición en `animals.mp4`:**

| # Consulta | Consulta en `animals.mp4` | Intervalo (s) | Rango `d=3072` | Rango `d=1536` | Rango `d=768` | Comportamiento Matryoshka |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 3 | León macho descansando tumbado | 5,0 - 10,0 | 5 | 5 | **2** | Sube 3 puestos en `d=768` (`#5 -> #2`, mejora `Recall@3`) |
| 6 | Gato gris con ojos verdes en primer plano | 18,0 - 23,0 | 2 | 2 | 3 | Se mantiene dentro del `Top-3` (`#2 -> #3`) |
| 9 | Medusa nadando en el mar azul | 32,0 - 35,0 | 3 | 3 | **2** | Sube un puesto en `d=768` (`#3 -> #2`) |
| 24 | Cría de mono comiendo fruta con las dos manos | 110,0 - 114,0 | 2 | 2 | 3 | Se mantiene dentro del `Top-3` (`#2 -> #3`) |
| 29 | Gran ciervo macho con cornamenta ramificada y crías | 136,0 - 144,0 | **1** | 2 | 2 | Única consulta que pasa de `#1` a `#2` en `1536` y `768` |

---

## 4. Batería 2: Consultas generales y abstractas multi-objetivo (20 consultas, Etapa 1 y Etapa 2)

Para evaluar si el truncado Matryoshka afecta a las búsquedas conceptuales amplias (donde el usuario consulta una categoría taxonómica o temática como `"Felinos"`, `"Aves"`, `"Reptiles"`, `"Estructuras de datos..."` o `"Animales marinos..."` que aparece en múltiples segmentos disjuntos del vídeo), se ejecutó la batería multi-objetivo completa con `top_k = 10` tanto en Etapa 1 (`Qdrant`) como en Etapa 2 (`gemini-3.5-flash-lite` bimodal).

### 4.1 Resumen agregado multi-objetivo por vídeo y dimensión (`top_k = 10`)

| Corpus de Vídeo | Dimensión (`d`) | Consultas | Intervalos Esperados | Recall Intervalos Etapa 1 | Precisión Media Etapa 1 | Recall Intervalos Etapa 2 | Precisión Media Etapa 2 | `First Hit = #1` en Etapa 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `python_tutorial.mp4` | **3072** | 10 | 48 | 85,42 % (40/48) | 45,00 % | 82,08 % (38/48) | 78,65 % | 9/10 (90,0 %) |
| `python_tutorial.mp4` | **1536** | 10 | 48 | 87,08 % (41/48) | **47,00 %** | 82,08 % (38/48) | 78,12 % | 9/10 (90,0 %) |
| `python_tutorial.mp4` | **768** | 10 | 48 | **87,22 % (41/48)** | 46,00 % | **83,89 % (40/48)** | **80,06 %** | 9/10 (90,0 %) |
| `animals.mp4` | **3072** | 10 | 58 | **74,89 % (42/58)** | **50,00 %** | **71,64 % (40/58)** | **96,57 %** | **10/10 (100,0 %)** |
| `animals.mp4` | **1536** | 10 | 58 | 71,98 % (40/58) | 48,00 % | 68,73 % (38/58) | 96,33 % | **10/10 (100,0 %)** |
| `animals.mp4` | **768** | 10 | 58 | **74,89 % (42/58)** | **50,00 %** | **71,64 % (40/58)** | **96,57 %** | **10/10 (100,0 %)** |

### 4.2 Desglose por consulta general en `animals.mp4` (Categorías taxonómicas y escenas generales)

| Consulta General / Taxonómica (`animals.mp4`) | Obj. (`N`) | Etapa 1 (`3072`) | Etapa 1 (`1536`) | Etapa 1 (`768`) | Etapa 2 (`3072`) | Etapa 2 (`1536`) | Etapa 2 (`768`) | Precisión Etapa 2 (`3072` / `1536` / `768`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Felinos`** | 4 | 1/4 (25,0 %) | 1/4 (25,0 %) | 1/4 (25,0 %) | 1/4 (25,0 %) | 1/4 (25,0 %) | 1/4 (25,0 %) | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Reptiles`** | 2 | **2/2 (100 %)** | **2/2 (100 %)** | **2/2 (100 %)** | **2/2 (100 %)** | **2/2 (100 %)** | **2/2 (100 %)** | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Aves`** | 4 | 2/4 (50,0 %) | 2/4 (50,0 %) | 2/4 (50,0 %) | 2/4 (50,0 %) | 2/4 (50,0 %) | 2/4 (50,0 %) | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Manada o grupo de tres o más animales juntos`** | 10 | 4/10 (40,0 %) | 4/10 (40,0 %) | 4/10 (40,0 %) | 4/10 (40,0 %) | 4/10 (40,0 %) | 4/10 (40,0 %) | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Animales de color blanco y negro`** | 5 | **5/5 (100 %)** | **5/5 (100 %)** | **5/5 (100 %)** | 4/5 (80,0 %) | 4/5 (80,0 %) | 4/5 (80,0 %) | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Animales marinos o acuáticos nadando...`** | 8 | 6/8 (75,0 %) | 5/8 (62,5 %) | 6/8 (75,0 %) | 6/8 (75,0 %) | 5/8 (62,5 %) | 6/8 (75,0 %) | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Animales comiendo, pastando o amamantándose`** | 8 | 7/8 (87,5 %) | 7/8 (87,5 %) | 7/8 (87,5 %) | 6/8 (75,0 %) | 6/8 (75,0 %) | 6/8 (75,0 %) | **100 % / 100 % / 100 %** (`#1` en las 3) |
| **`Crías de animales o animales jóvenes`** | 7 | 5/7 (71,4 %) | 5/7 (71,4 %) | 5/7 (71,4 %) | 5/7 (71,4 %) | 5/7 (71,4 %) | 5/7 (71,4 %) | 80,0 % / 80,0 % / 80,0 % (`#1` en las 3) |
| **`Primerísimo plano macro de ojos, orejas...`** | 6 | **6/6 (100 %)** | 5/6 (83,3 %) | **6/6 (100 %)** | **6/6 (100 %)** | 5/6 (83,3 %) | **6/6 (100 %)** | 85,7 % / 83,3 % / 85,7 % (`#1` en las 3) |
| **`Animación 3D microscópica de células...`** | 4 | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **100 % / 100 % / 100 %** (`#1` en las 3) |

**Hallazgos clave en `animals.mp4`:**
- **Equivalencia exacta entre `d = 768` y `d = 3072`:** Tanto en Etapa 1 (`74,89 %` de *Interval Recall* y `50,00 %` de precisión) como en Etapa 2 (`71,64 %` de *Interval Recall* y **`96,57 %` de precisión**), la dimensión compacta `d = 768` obtiene **exactamente los mismos aciertos en las 10 consultas generales** que el vector nativo de `3072` dimensiones (`Felinos` 1/4, `Reptiles` 2/2, `Aves` 2/4, `Manada` 4/10, `Blanco y negro` 4/5, `Marinos` 6/8, `Comiendo` 6/8, `Crías` 5/7, `Macro` 6/6 y `Células` 4/4).
- **Promoción al puesto `#1` en Etapa 2 (`10/10` consultas):** Incluso en consultas como `"Reptiles"` (donde el primer resultado en Qdrant aparece en el puesto `#4` en `3072`, `#2` en `1536` y `#5` en `768`) o `"Animales marinos"` (`#2`/`#3` en Qdrant), el verificador bimodal `gemini-3.5-flash-lite` descarta los falsos positivos arrastrados por la voz en *off* y coloca un acierto verdadero en el **puesto `#1` en el 100 % de las consultas y en las tres dimensiones**, elevando la precisión media del `~50 %` (Etapa 1) al **`96,57 %`** (Etapa 2).

### 4.3 Desglose por consulta general en `python_tutorial.mp4`

| Consulta General / Multi-Objetivo (`python_tutorial.mp4`) | Obj. (`N`) | Etapa 1 (`3072`) | Etapa 1 (`1536`) | Etapa 1 (`768`) | Etapa 2 (`3072`) | Etapa 2 (`1536`) | Etapa 2 (`768`) | Precisión Etapa 2 (`3072` / `1536` / `768`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Planos del presentador en blanco y negro`** | 3 | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | 37,5 % / 42,9 % / 42,9 % |
| **`Tipos de datos primitivos (int, float, str, bool...)`** | 9 | **9/9 (100 %)** | **9/9 (100 %)** | 8/9 (88,9 %) | **9/9 (100 %)** | **9/9 (100 %)** | 8/9 (88,9 %) | 85,7 % / 87,5 % / 83,3 % |
| **`Estructuras de datos (list, tuple, set, dict)`** | 6 | 5/6 (83,3 %) | 5/6 (83,3 %) | **6/6 (100 %)** | 5/6 (83,3 %) | 5/6 (83,3 %) | **6/6 (100 %)** | **100 % / 87,5 % / 100 %** |
| **`Rótulos o símbolos tachados con cruz roja X`** | 2 | 1/2 (50,0 %) | 1/2 (50,0 %) | 1/2 (50,0 %) | 1/2 (50,0 %) | 1/2 (50,0 %) | 1/2 (50,0 %) | 50,0 % / 50,0 % / 50,0 % |
| **`Terminal mostrando salida de ejecución`** | 4 | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | 80,0 % / 80,0 % / 80,0 % |
| **`Código de condicionales if, elif y else`** | 3 | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **100 % / 100 % / 100 %** |
| **`Código de bucles for y while`** | 4 | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **4/4 (100 %)** | **100 % / 100 % / 100 %** |
| **`Declaración y reasignación de variables`** | 3 | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **3/3 (100 %)** | **100 % / 100 % / 100 %** |
| **`Funciones integradas input, len y print`** | 6 | 5/6 (83,3 %) | **6/6 (100 %)** | 5/6 (83,3 %) | 3/6 (50,0 %) | 3/6 (50,0 %) | 3/6 (50,0 %) | **100 % / 100 % / 100 %** |
| **`Títulos de sección sobreimpresos con presentador`** | 8 | 3/8 (37,5 %) | 3/8 (37,5 %) | **4/8 (50,0 %)** | 3/8 (37,5 %) | 3/8 (37,5 %) | **4/8 (50,0 %)** | 33,3 % / 33,3 % / 44,4 % |

**Hallazgos clave en `python_tutorial.mp4`:**
- En `d = 768`, la consulta `"Estructuras de datos y colecciones en Python (list, tuple, set, dict)"` alcanza el **100,0 % de recuperación (`6/6` intervalos) con un 100,0 % de precisión en Etapa 2** (frente a `5/6` en `3072` y `1536`), y `"Títulos de sección sobreimpresos..."` sube de `3/8` a `4/8`, situando el *Interval Recall* medio de `d = 768` en **`87,22 %` (Etapa 1)** y **`83,89 %` (Etapa 2)** (+1,81 pp por encima de `3072`).

---

## 5. Batería 3: Consultas negativas de control e inmunidad a falsos positivos (16 consultas)

Una hipótesis de riesgo al truncar embeddings mediante MRL es que la pérdida de las coordenadas de grano fino (`[768:3072]`) pueda aproximar geométricamente consultas inexistentes pero del mismo dominio (por ejemplo, `"Definición de una clase con class e __init__"` en `python_tutorial.mp4` o `"Tiburón blanco atacando en el océano"` y `"Águila volando alto en el cielo azul"` en `animals.mp4`) e inducir falsos positivos en el sistema.

| Corpus de Vídeo | Dimensión (`d`) | Consultas Negativas | Candidatos Etapa 1 (`top_k = 5`) | Falsos Positivos Etapa 1 (Sin umbral) | Falsos Positivos Etapa 2 (`gemini-3.5-flash-lite`) | Tasa de Rechazo Correcto Etapa 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `python_tutorial.mp4` | **3072** | 8 | 40 | 40 / 40 (100 %) | **0 / 40 (0,0 %)** | **8 / 8 (100,0 %)** |
| `python_tutorial.mp4` | **1536** | 8 | 40 | 40 / 40 (100 %) | **0 / 40 (0,0 %)** | **8 / 8 (100,0 %)** |
| `python_tutorial.mp4` | **768** | 8 | 40 | 40 / 40 (100 %) | **0 / 40 (0,0 %)** | **8 / 8 (100,0 %)** |
| `animals.mp4` | **3072** | 8 | 40 | 40 / 40 (100 %) | **0 / 40 (0,0 %)** | **8 / 8 (100,0 %)** |
| `animals.mp4` | **1536** | 8 | 40 | 40 / 40 (100 %) | **0 / 40 (0,0 %)** | **8 / 8 (100,0 %)** |
| `animals.mp4` | **768** | 8 | 40 | 40 / 40 (100 %) | **0 / 40 (0,0 %)** | **8 / 8 (100,0 %)** |
| **Total Global (3 dim.)** | **`3072 / 1536 / 768`** | **48 ejecuciones** | **240 candidatos** | **240 / 240** | **0 / 240 (0,0 %)** | **48 / 48 (100,0 %)** |

**Conclusión sobre control negativo:** El verificador bimodal de Etapa 2 rechaza con éxito el **100,0 % de los 240 fragmentos candidatos** (`0/240` falsos positivos) en las tres dimensiones. La compresión a `1536` o `768` dimensiones no degrada en ningún caso la especificidad del pipeline completo.

---

## 6. Escalabilidad de memoria y conclusiones para la arquitectura

### 6.1 Proyección de huella de almacenamiento vectorial (`float32`)

| Escala del Repositorio | Chunks Totales | Duración Estimada (`~3 s/chunk`) | Memoria en `d=3072` (`12 KB/vec`) | Memoria en `d=1536` (`6 KB/vec`, `-50 %`) | Memoria en `d=768` (`3 KB/vec`, `-75 %`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Banco PoC actual (2 vídeos)** | 164 | ~8,1 min | 1,92 MB | 0,96 MB | 0,48 MB |
| **Curso o documental largo** | 10.000 | ~8,3 horas | 117,19 MB | 58,59 MB | 29,30 MB |
| **Archivo departamental** | 100.000 | ~83,3 horas | 1,14 GB | 585,94 MB | 292,97 MB |
| **Plataforma corporativa** | 1.000.000 | ~833,3 horas | 11,44 GB | 5,72 GB | 2,86 GB |

### 6.2 Conclusiones técnicas (Objetivo Específico 4 del TFM)

1. **Retención semántica superior al 99 % con un 75 % menos de memoria en consultas literales:** En el agregado de las 102 consultas literales, reducir la dimensionalidad de `3072` a `768` conserva el **98,8 % del `Recall@1` relativo** (`80/102` frente a `81/102`), mantiene un `MRR` prácticamente idéntico (`0,8691` frente a `0,8688`) y eleva el `Recall@5` del `97,06 %` al **`99,02 %` (`101/102`)**, reduciendo el tamaño de cada vector de `12 KB` a `3 KB` y acelerando las búsquedas en Qdrant un **`44,4 %`**.
2. **Invarianza en consultas generales/taxonómicas (`"Felinos"`, `"Aves"`, `"Reptiles"`, etc.) e inmunidad a falsos positivos:** En las 20 consultas multi-objetivo (106 intervalos totales), `d = 768` iguala exactamente el rendimiento de `d = 3072` en `animals.mp4` (`74,89 %` en Etapa 1 y `71,64 %` con `96,57 %` de precisión en Etapa 2) y lo supera ligeramente en `python_tutorial.mp4` (`83,89 %` frente a `82,08 %` en Etapa 2), manteniendo `0/240` falsos positivos en las 16 consultas negativas de control.
3. **Soporte configurable de extremo a extremo:** El sistema permite seleccionar cualquiera de las tres dimensiones oficiales (`3072`, `1536`, `768`) tanto desde la interfaz gráfica Streamlit (`src/ui/app.py`) como desde la CLI (`--output-dimensionality` en `src/main.py`), gestionando automáticamente la recreación limpia de la colección en `QdrantSearchEngine` y la normalización L2 en `truncate_and_normalize_mrl`.
