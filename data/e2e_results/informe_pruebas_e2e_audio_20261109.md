# Informe comparativo de pruebas E2E: Pipeline Visual vs. Pipeline Multimodal (Visual + Audio Whisper)

Fechas de ejecución comparadas:
- **Línea base visual (`2026-11-06`):**
  - `animals.mp4` (243,0 s): `audit/run_20261106_110324/`
  - `python_tutorial.mp4` (242,5 s): `audit/run_20261106_134026/`
- **Nueva implementación multimodal con transcripción de audio (`2026-11-09`):**
  - `python_tutorial.mp4` (242,5 s): `audit/run_20261109_144503/`
  - `animals.mp4` (243,0 s): `audit/run_20261109_145448/`

Copias versionadas de los informes CSV en `data/e2e_results/`:
- `python_tutorial.mp4` (Visual, `2026-11-06`): [`python_e2e_report.csv`](python_e2e_report.csv), [`python_e2e_multi_target_report.csv`](python_e2e_multi_target_report.csv), [`python_e2e_negative_report.csv`](python_e2e_negative_report.csv)
- `python_tutorial.mp4` (Visual + Audio, `2026-11-09`): [`python_e2e_report_20261109.csv`](python_e2e_report_20261109.csv), [`python_e2e_multi_target_report_20261109.csv`](python_e2e_multi_target_report_20261109.csv), [`python_e2e_negative_report_20261109.csv`](python_e2e_negative_report_20261109.csv)
- `animals.mp4` (Visual, `2026-11-06`): [`e2e_report_20261106.csv`](e2e_report_20261106.csv), [`e2e_multi_target_report_20261105.csv`](e2e_multi_target_report_20261105.csv), [`e2e_negative_report_20261105.csv`](e2e_negative_report_20261105.csv)
- `animals.mp4` (Visual + Audio, `2026-11-09`): [`e2e_report_20261109.csv`](e2e_report_20261109.csv), [`e2e_multi_target_report_20261109.csv`](e2e_multi_target_report_20261109.csv), [`e2e_negative_report_20261109.csv`](e2e_negative_report_20261109.csv)

---

## 1. Objetivo del informe

Evaluar empíricamente el impacto de la integración del canal auditivo (`ffmpeg` + `openai-whisper` + fusión temprana en `gemini-embedding-2-preview` + reordenación bimodal en `gemini-3.5-flash-lite`) frente a la arquitectura puramente visual documentada en [`informe_pruebas_e2e.md`](informe_pruebas_e2e.md).

Ambos vídeos del banco de pruebas representan escenarios acústicos opuestos:
1. **`python_tutorial.mp4` (Locución acoplada a la imagen):** el presentador explica verbalmente los mismos conceptos, estructuras de datos y fragmentos de código que muestra en pantalla o introduce conceptos hablados cuyo rótulo visual es mínimo.
2. **`animals.mp4` (Locución divulgativa sobre vídeo de archivo / B-roll):** la narración expone conceptos generales de biología animal (células eucariotas, nutrición heterótrofa, reproducción) mientras la imagen muestra planos cortos de animales variados que no se nombran en la locución, salvo en las cabeceras con texto sobreimpreso (`REINO ANIMALIA`, `ECOLOGÍA VERDE`).

---

## 2. Resumen ejecutivo de resultados

### 2.1 Comparativa global en consultas literales (`top_k = 5`)

| Vídeo | Etapa | Métrica | Visual (`2026-11-06`) | Visual + Audio (`2026-11-09`) | Variación |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `python_tutorial.mp4` (56 consultas) | **Etapa 1 (Retrieval)** | **Recall@1** | 67,9 % (38/56) | **85,7 % (48/56)** | **+17,8 pp (+10 aciertos)** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 1 (Retrieval)** | **Recall@3** | 87,5 % (49/56) | **92,9 % (52/56)** | **+5,4 pp (+3 aciertos)** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 1 (Retrieval)** | **Recall@5** | 91,1 % (51/56) | **94,6 % (53/56)** | **+3,5 pp (+2 aciertos)** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 1 (Retrieval)** | **MRR** | 0,779 | **0,897** | **+0,118** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 2 (Juez Bimodal)** | **Recall@1** | 80,4 % (45/56) | **89,3 % (50/56)** | **+8,9 pp (+5 aciertos)** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 2 (Juez Bimodal)** | **Recall@3** | 89,3 % (50/56) | **92,9 % (52/56)** | **+3,6 pp (+2 aciertos)** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 2 (Juez Bimodal)** | **Recall@5** | 89,3 % (50/56) | **92,9 % (52/56)** | **+3,6 pp (+2 aciertos)** |
| `python_tutorial.mp4` (56 consultas) | **Etapa 2 (Juez Bimodal)** | **MRR** | 0,848 | **0,911** | **+0,063** |
| `animals.mp4` (46 consultas) | **Etapa 1 (Retrieval)** | **Recall@1** | 80,4 % (37/46)* | 71,7 % (33/46) | -8,7 pp |
| `animals.mp4` (46 consultas) | **Etapa 1 (Retrieval)** | **Recall@3** | 95,7 % (44/46) | 89,1 % (41/46) | -6,6 pp |
| `animals.mp4` (46 consultas) | **Etapa 1 (Retrieval)** | **Recall@5** | 97,8 % (45/46) | 95,7 % (44/46) | -2,1 pp |
| `animals.mp4` (46 consultas) | **Etapa 1 (Retrieval)** | **MRR** | 0,891 | 0,813 | -0,078 |
| `animals.mp4` (46 consultas) | **Etapa 2 (Juez Bimodal)** | **Recall@1** | 89,1 % (41/46) | **93,5 % (43/46)** | **+4,4 pp (+2 aciertos)** |
| `animals.mp4` (46 consultas) | **Etapa 2 (Juez Bimodal)** | **Recall@3** | 93,5 % (43/46) | **95,7 % (44/46)** | **+2,2 pp (+1 acierto)** |
| `animals.mp4` (46 consultas) | **Etapa 2 (Juez Bimodal)** | **Recall@5** | 93,5 % (43/46) | **97,8 % (45/46)** | **+4,3 pp (+2 aciertos)** |
| `animals.mp4` (46 consultas) | **Etapa 2 (Juez Bimodal)** | **MRR** | 0,913 | **0,951** | **+0,038** |

*\*Nota: en `animals.mp4`, el informe previo registró `80,4 % (37/46)` en la tabla resumen de `run_20261106_110324` y `84,8 % (39/46)` en el CSV exportado [`e2e_report_20261106.csv`](e2e_report_20261106.csv).*

### 2.2 Estado de la suite de tests y calidad de código

| Verificación | Comando | Resultado |
| :--- | :--- | :--- |
| Suite unitaria completa (offline) | `./env_tfm/bin/pytest` | **133 passed, 14 deselected** (1,99 s) |
| Linter estático | `./env_tfm/bin/ruff check src tests app.py` | **All checks passed** |
| Tipado estático general | `./env_tfm/bin/python -m mypy --explicit-package-bases src tests app.py` | **Success: 0 issues (41 files)** |
| Tipado estático estricto (`src`) | `./env_tfm/bin/python -m mypy --strict --explicit-package-bases src` | **Success: 0 issues (20 files)** |
| Suite E2E `python_tutorial.mp4` | `./env_tfm/bin/pytest tests/e2e/test_python_e2e.py -m e2e` | **7 passed in 568,91 s (9 min 28 s)** |
| Suite E2E `animals.mp4` | `./env_tfm/bin/pytest tests/e2e/test_animals_e2e.py -m e2e` | **7 passed in 463,75 s (7 min 43 s)** |

---

## 3. Análisis detallado en `python_tutorial.mp4` (`run_20261109_144503`)

### 3.1 Indexación multimodal

- **Segmentación visual (`threshold = 0,6`, `max_shot_duration = 5,0 s`):** 82 chunks detectados en `25,38 s`.
- **Transcripción y alineación temporal (`Whisper base`, idioma `es`):** completada en **`10,96 s`**, enriqueciendo **81 de los 82 chunks (`98,8 %`)** con locución transcrita (el único fragmento sin voz corresponde a la introducción muda de `0,0 - 0,7 s`).
- **Vectorización multimodal (`gemini-embedding-2-preview`):** cada uno de los 81 fragmentos con voz se incrustó combinando en una misma llamada sus 3 fotogramas JPEG (`inicio`, `medio`, `fin`) y la parte textual `Transcripción de audio del fragmento: ...`.

### 3.2 Impacto en las 56 consultas literales

La incorporación del audio produce un salto en ambas etapas del pipeline sobre `python_tutorial.mp4`:
1. **Etapa 1 (Recuperación vectorial en Qdrant):** el `Recall@1` sube del **67,9 % (38/56)** al **85,7 % (48/56)** (**+17,8 puntos porcentuales**), y el `MRR` sube de **0,779** a **0,897**. Diez consultas que antes dependían del reranker o fallaban en la primera etapa ahora se colocan directamente en el **puesto 1** gracias a la coincidencia entre la pregunta y la explicación hablada del instructor.
2. **Etapa 2 (Reordenación bimodal con `gemini-3.5-flash-lite`):** el `Recall@1` sube del **80,4 % (45/56)** al **89,3 % (50/56)** (**+8,9 puntos porcentuales**), el `Recall@5` sube del **89,3 % (50/56)** al **92,9 % (52/56)**, y el `MRR` alcanza **0,911**.

#### Detalle de las 15 consultas literales que cambian de posición en `python_tutorial.mp4`

| # | Consulta | Intervalo (s) | Recup. Visual | Rerank Visual | Recup. Audio+Vis | Rerank Audio+Vis | Efecto de la fusión bimodal |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | Presentador en B/N saludando con dos manos | 0,0-0,7 | 3 | 1 | 4 | - | Único chunk mudo (`0,0-0,7 s`); el juez prioriza otros planos B/N con voz |
| 6 | Rótulo `int` para explicar números enteros | 25,0-27,9 | 2 | 1 | **1** | **1** | Sube al puesto 1 desde Etapa 1 (`#2 -> #1`) |
| 8 | Tipo de cadenas mostrando únicamente el rótulo `str` | 30,5-33,5 | 3 | 2 | **1** | **1** | Sube al puesto 1 en Etapa 1 (`#3 -> #1`) y Etapa 2 (`#2 -> #1`) |
| 13 | Números complejos con el rótulo `complex` | 47,5-49,8 | 5 | 2 | **2** | **2** | Mejora 3 puestos en Etapa 1 (`#5 -> #2`) |
| 22 | Título `Variables` en letras blancas grandes | 96,1-100,2 | 2 | 2 | **1** | **1** | Sube al puesto 1 en Etapa 1 y Etapa 2 (`#2 -> #1`) |
| 27 | Gesto de precisión tras mostrar reasignación | 122,3-127,3 | - | - | **1** | - | Etapa 1 rescata un fallo total directamente al puesto 1 (`- -> #1`) |
| 28 | Rótulo `snake_case` para nombrar variables | 127,3-132,8 | 2 | 1 | **1** | **1** | Sube al puesto 1 desde Etapa 1 (`#2 -> #1`) |
| 34 | Instrucción `print('Hola, Python!')` con rótulo `print` | 156,7-158,7 | - | 1 | **1** | **1** | Etapa 1 rescata un fallo al puesto 1 (`- -> #1`) |
| 36 | Comando `python3 hello_python.py` en terminal | 161,6-166,5 | 2 | 1 | **1** | **1** | Sube al puesto 1 desde Etapa 1 (`#2 -> #1`) |
| 42 | `my_age = 37` y cabecera `if my_age >= 18:` | 185,0-191,0 | 4 | 1 | **1** | **1** | Sube 3 puestos en Etapa 1 (`#4 -> #1`) |
| 44 | Presentador enmarca llaves `{ }` aún sin tachar | 197,6-201,8 | 1 | - | **1** | **1** | El juez bimodal ya no descarta el acierto (`- -> #1`) |
| 49 | Primera salida del 0 al 4 tras ejecutar `range(5)` | 224,4-225,3 | - | - | - | **1** | Rescata un fallo total de ambas etapas al puesto 1 (`- -> #1`) |
| 51 | Terminal muestra números del 1 al 5 tras `my_list` | 228,1-230,2 | 2 | 1 | **1** | **1** | Sube al puesto 1 desde Etapa 1 (`#2 -> #1`) |
| 52 | `while index < 5` con rótulo `while` sobreimpreso | 230,2-232,5 | 2 | 2 | **1** | **1** | Sube al puesto 1 en Etapa 1 y Etapa 2 (`#2 -> #1`) |
| 54 | Salida en terminal del 0 al 4 del bucle `while` | 236,0-238,3 | 1 | - | **1** | **1** | El juez bimodal confirma el acierto con la locución (`- -> #1`) |

### 3.3 Consultas abstractas multi-objetivo en `python_tutorial.mp4` (10 consultas, `top_k = 10`)

Comparativa entre `run_20261106_134026` (Visual) y `run_20261109_144503` (Visual + Audio, [`python_e2e_multi_target_report_20261109.csv`](python_e2e_multi_target_report_20261109.csv)):

| # | Consulta abstracta | Planos reales | Recall Recup. (Visual -> Audio) | Precisión Recup. (Visual -> Audio) | Recall Rerank (Visual -> Audio) | Precisión Rerank (Visual -> Audio) | 1.er acierto Rerank |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Planos del presentador en blanco y negro` | 3 | 100,0 % -> **100,0 %** | 40,0 % -> 30,0 % | 100,0 % -> **100,0 %** | 44,4 % -> 37,5 % | 1 -> **1** |
| 2 | `Tipos de datos primitivos (int, float, str, bool, None, complex)` | 9 | 66,7 % -> **100,0 %** | 40,0 % -> **60,0 %** | 66,7 % -> **100,0 %** | 66,7 % -> **85,7 %** | 1 -> **1** |
| 3 | `Estructuras de datos y colecciones (list, tuple, set, dict)` | 6 | 100,0 % -> 83,3 % | 70,0 % -> **80,0 %** | 100,0 % -> 83,3 % | 100,0 % -> **100,0 %** | 1 -> **1** |
| 4 | `Rótulos o símbolos tachados con una cruz roja X` | 2 | 100,0 % -> 50,0 % | 20,0 % -> 20,0 % | 50,0 % -> 50,0 % | 50,0 % -> 50,0 % | 1 -> **1** |
| 5 | `Terminal o consola mostrando la salida de ejecución` | 4 | 75,0 % -> **100,0 %** | 40,0 % -> **60,0 %** | 75,0 % -> **100,0 %** | 75,0 % -> **80,0 %** | 2 -> **1** |
| 6 | `Código de condicionales if, elif y else en VS Code` | 3 | 100,0 % -> **100,0 %** | 50,0 % -> 50,0 % | 100,0 % -> **100,0 %** | 100,0 % -> **100,0 %** | 1 -> **1** |
| 7 | `Código de bucles for y while en VS Code` | 4 | 100,0 % -> **100,0 %** | 40,0 % -> 40,0 % | 100,0 % -> **100,0 %** | 100,0 % -> **100,0 %** | 1 -> **1** |
| 8 | `Declaración y reasignación de variables en VS Code` | 3 | 66,7 % -> **100,0 %** | 50,0 % -> 50,0 % | 66,7 % -> **100,0 %** | 100,0 % -> **100,0 %** | 1 -> **1** |
| 9 | `Uso de funciones integradas input, len y print` | 6 | 66,7 % -> **83,3 %** | 40,0 % -> **50,0 %** | 50,0 % -> 50,0 % | 100,0 % -> **100,0 %** | 1 -> **1** |
| 10 | `Títulos de sección sobreimpresos en letras blancas` | 8 | 75,0 % -> 37,5 % | 60,0 % -> 30,0 % | 75,0 % -> 37,5 % | 66,7 % -> 33,3 % | 1 -> 3 |
| **Media** | - | **4,8** | **85,0 % -> 85,4 %** | **45,0 % -> 47,0 %** | **78,3 % -> 82,1 %** | **80,3 % -> 78,7 %** | **1,1 -> 1,2** |

Hallazgos clave en la batería multi-objetivo de `python_tutorial.mp4`:
- **La cobertura media tras el juez bimodal (`Interval Recall`) sube del 78,3 % al 82,1 % (+3,8 puntos porcentuales).**
- **3 categorías técnicas suben hasta el 100,0 % de cobertura completa:**
  - `Tipos de datos primitivos (int, float, str, bool, None, complex)`: sube de `6/9 (66,7 %)` a **`9/9 (100,0 %)`** y su precisión tras rerank sube del `66,7 %` al **`85,7 %`**.
  - `Terminal o consola mostrando la salida de ejecución`: sube de `3/4 (75,0 %)` a **`4/4 (100,0 %)`**, su precisión sube del `75,0 %` al **`80,0 %`** y el primer acierto sube del puesto `2` al puesto **`1`**.
  - `Declaración y reasignación de variables en VS Code`: sube de `2/3 (66,7 %)` a **`3/3 (100,0 %)`** con un **`100,0 %` de precisión**.

### 3.4 Consultas negativas en `python_tutorial.mp4` (8 consultas, `top_k = 5`)

En [`python_e2e_negative_report_20261109.csv`](python_e2e_negative_report_20261109.csv), el juez bimodal mantiene el **100,0 % de tasa de rechazo (`8/8` consultas rechazadas, `0/40` falsos positivos)** frente a distractores de programación en Python (`class / __init__`, `matplotlib / pandas`, `pip install`, `PyTorch`) y fuera de dominio.

---

## 4. Análisis detallado en `animals.mp4` (`run_20261109_145448`)

### 4.1 Indexación multimodal

- **Segmentación visual (`threshold = 0,5`, `max_shot_duration = 8,0 s`):** 82 chunks detectados en `14,15 s`.
- **Transcripción y alineación temporal (`Whisper base`, idioma `es`):** completada en **`9,45 s`**, enriqueciendo **82 de los 82 chunks (`100,0 %`)** con la locución divulgativa del vídeo.

### 4.2 Impacto en las 46 consultas literales

En `animals.mp4` se observa con nitidez la complementariedad entre ambas etapas cuando el audio es una locución en *off* desacoplada de las imágenes de archivo (*B-roll*):
- En la **Etapa 1 (Retrieval puro)**, la inclusión del discurso biológico general en el embedding reduce ligeramente la similitud coseno frente a consultas puramente visuales de animales concretos (`Recall@1` de `80,4 %` a `71,7 %`, manteniendo un `Recall@5` del `95,7 %`).
- En la **Etapa 2 (Juez Bimodal)**, `gemini-3.5-flash-lite` inspecciona tanto el collage visual de 3 fotogramas como la transcripción y consigue el **mejor resultado histórico del proyecto sobre `animals.mp4`**:
  - **`Recall@1` sube del 89,1 % (41/46) al 93,5 % (43/46) (+4,4 puntos porcentuales).**
  - **`Recall@3` sube del 93,5 % (43/46) al 95,7 % (44/46) (+2,2 puntos porcentuales).**
  - **`Recall@5` sube del 93,5 % (43/46) al 97,8 % (45/46) (+4,3 puntos porcentuales, acertando 45 de las 46 consultas).**
  - **`MRR` sube de 0,913 a 0,951 (+0,038).**

#### Consultas literales de `animals.mp4` rescatadas o promovidas por el juez bimodal

| # | Consulta | Intervalo (s) | Rerank Visual (`2026-11-06`) | Rerank Audio+Vis (`2026-11-09`) | Causa de la mejora |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 2 | Lémures de cola anillada con rótulo `REINO ANIMALIA` | 1,0-5,0 | 3 (o `-` en CSV) | **1** | La locución pronuncia *"Reino Animalia"* coincidiendo con el rótulo |
| 4 | Cabecera con el texto `ECOLOGÍA VERDE` | 10,0-14,0 | - | **1** | La locución y la cabecera coinciden; rescata un fallo al puesto 1 |
| 14 | Ave rapaz de plumaje marrón y pecho blanco | 52,0-59,0 | - (en CSV) / 1 | **1** | Confirmado en el puesto 1 con `hit_rank = 1` en ambas etapas |
| 31 | Animación 3D de glóbulos rojos en vaso sanguíneo | 158,0-170,0 | 2 | **1** | La narración sobre células eucariotas refuerza la imagen 3D (`#2 -> #1`) |

### 4.3 Consultas abstractas multi-objetivo y negativas en `animals.mp4`

- **Consultas multi-objetivo ([`e2e_multi_target_report_20261109.csv`](e2e_multi_target_report_20261109.csv)):** dado que las 10 consultas buscan categorías visuales amplias (`Felinos`, `Reptiles`, `Aves`, `Manadas`) a lo largo de múltiples planos de archivo mientras el narrador habla de teoría celular o reproducción, el `Interval Recall` medio en Etapa 1 baja del `85,4 %` al `67,0 %` (`55,7 %` tras el juez), si bien el juez bimodal mantiene una **precisión media del 88,3 %** (**100,0 % de precisión** en 8 de las 9 consultas con resultados: `Felinos`, `Aves`, `Manada`, `Blanco y negro`, `Animales marinos`, `Animales comiendo`, `Crías` y `Animación 3D de células`). Este comportamiento justifica disponer de la opción `--no-audio` (y de la casilla en la interfaz Streamlit) cuando se indexan bancos de metraje mudo o con voz en *off* no descriptiva.
- **Consultas negativas ([`e2e_negative_report_20261109.csv`](e2e_negative_report_20261109.csv)):** el juez bimodal mantiene el **100,0 % de rechazo (`8/8` consultas rechazadas, `0/40` falsos positivos)**.

---

## Apéndice A: Comparativa por consulta literal en `animals.mp4` (46 consultas)

Comparativa directa entre [`e2e_report_20261106.csv`](e2e_report_20261106.csv) (`Visual`) y [`e2e_report_20261109.csv`](e2e_report_20261109.csv) (`Audio+Vis`). El guion (`-`) indica ausencia de acierto en el top-5.

| # | Consulta | Intervalo (s) | Recup. Visual | Rerank Visual | Recup. Audio+Vis | Rerank Audio+Vis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | ¿En qué segundo aparece un elefante caminando en un recinto? | 0-1 | 1 | 1 | 1 | 1 |
| 2 | ¿En qué momento aparecen lémures de cola anillada con el rótulo de REINO ANIMALIA? | 1-5 | - | - | **1** | **1** |
| 3 | ¿Dónde y cuándo se muestra un león macho descansando tumbado? | 5-10 | 1 | 1 | - | 1 |
| 4 | ¿En qué intervalo temporal aparece la cabecera con el texto ECOLOGÍA VERDE? | 10-14 | 1 | - | 3 | **1** |
| 5 | ¿Cuándo se observa a tres elefantes incluyendo una cría caminando entre la hierba? | 14-18 | 1 | 1 | 1 | 1 |
| 6 | ¿En qué segundo aparece un gato gris con ojos verdes en primer plano? | 18-23 | 1 | 1 | 2 | 1 |
| 7 | ¿En qué momento aparece el ojo de un camaleón en primer plano macro? | 23-29 | 1 | 1 | 1 | 1 |
| 8 | ¿Cuándo se muestra a un grupo de ciervas descansando en la nieve? | 29-32 | 2 | 1 | 4 | 1 |
| 9 | ¿En qué intervalo se ve una medusa nadando en el mar azul? | 32-35 | 2 | 1 | 3 | 1 |
| 10 | ¿Dónde aparece una leona con un collar de rastreo caminando? | 35-39 | 1 | 1 | 1 | 1 |
| 11 | ¿En qué segundos se ven tres cisnes blancos en un lago al amanecer? | 39-45 | 1 | 1 | 1 | 1 |
| 12 | ¿Cuándo aparece un ternero negro con la cara blanca en primer plano? | 45-49 | 1 | 1 | 2 | 1 |
| 13 | ¿En qué momento aparece una vaca Hereford marrón y blanca junto a su cría? | 49-52 | 1 | 1 | 2 | 1 |
| 14 | ¿En qué parte del vídeo se ve un ave rapaz de plumaje marrón y pecho blanco en primer plano? | 52-59 | 1 | - | 1 | **1** |
| 15 | ¿En qué segundos aparece una nutria marina flotando de espaldas en el agua? | 59-64 | 2 | 2 | 2 | 2 |
| 16 | ¿Cuándo se muestra una animación 3D de los orgánulos de una célula animal? | 64-76 | 1 | 1 | 1 | 1 |
| 17 | ¿En qué momento se observa a un caballo pastando hierba verde de cerca? | 76-81 | 1 | 1 | 1 | 1 |
| 18 | ¿Cuándo aparece un ternero mamando de la ubre de una vaca? | 81-86 | 1 | 1 | 1 | 1 |
| 19 | ¿En qué segundos aparece una ardilla comiendo plátanos sobre una tabla de madera? | 86-90 | 1 | 1 | 1 | 1 |
| 20 | ¿Dónde y en qué segundo se muestra una oveja blanca con etiquetas amarillas pastando? | 90-96 | 1 | 1 | 1 | 1 |
| 21 | ¿En qué intervalo temporal corre una manada de ciervos por un bosque nevado? | 96-102 | 1 | 1 | 1 | 1 |
| 22 | ¿Cuándo aparece una cría de foca moteada recostada en la arena de la playa? | 102-106 | 1 | 1 | 1 | 1 |
| 23 | ¿En qué momento aparecen tres pingüinos en una orilla rocosa? | 106-110 | 1 | 1 | 1 | 1 |
| 24 | ¿Cuándo aparece una cría de mono comiendo fruta con las dos manos? | 110-114 | 2 | 1 | 2 | 1 |
| 25 | ¿En qué segundos se ven cebras comiendo pasto en una llanura árida? | 114-120 | 1 | 1 | 1 | 1 |
| 26 | ¿En qué intervalo aparece un gallo o pollo blanco con cresta roja en primer plano? | 120-124 | 1 | 1 | 1 | 1 |
| 27 | ¿Cuándo se muestra a tres osos pardos juntos en un prado verde? | 124-130 | 1 | 1 | 1 | 1 |
| 28 | ¿En qué momento aparece el primer plano de la oreja peluda de un animal escuchando? | 130-136 | 1 | 1 | 1 | 1 |
| 29 | ¿En qué segundos se ve un gran ciervo macho con cornamenta ramificada y crías moteadas? | 136-144 | 2 | 1 | **1** | 1 |
| 30 | ¿Cuándo aparecen tres monos sentados juntos acicalándose y desparasitándose? | 144-158 | 1 | 1 | 1 | 1 |
| 31 | ¿En qué parte se observa una animación 3D de glóbulos rojos dentro de un vaso sanguíneo? | 158-170 | 2 | 2 | **1** | **1** |
| 32 | ¿Cuándo se muestra una animación azul de células fluyendo en espiral? | 170-174 | 1 | 1 | 1 | 1 |
| 33 | ¿En qué intervalo temporal se ven eritrocitos en primer plano dentro de una arteria roja? | 174-180 | 1 | 1 | 4 | 4 |
| 34 | ¿En qué segundo aparece el primerísimo plano del hocico de un ciervo respirando? | 180-186 | 1 | 1 | 1 | 1 |
| 35 | ¿Cuándo se ve a un perro marrón jadeando con la lengua fuera? | 186-193 | 1 | 1 | 1 | 1 |
| 36 | ¿En qué momento aparece el ojo verde de un gato atigrado de perfil? | 193-196 | 1 | 1 | 1 | 1 |
| 37 | ¿En qué segundos se ve a una raya marina o mantarraya nadando en el agua? | 196-198 | 1 | 1 | 1 | 1 |
| 38 | ¿Cuándo aparece una araña saltarina en macro con sus múltiples ojos negros? | 198-201 | 1 | 1 | 1 | 1 |
| 39 | ¿En qué intervalo se muestra el ojo dorado y la cabeza de una rana de piel húmeda? | 201-208 | 1 | 1 | 1 | 1 |
| 40 | ¿Cuándo aparece un caballo marrón con una mancha blanca en la frente? | 208-212 | 1 | 1 | - | 1 |
| 41 | ¿En qué segundos aparece una medusa gigante azul moteada con un buceador? | 212-217 | 1 | 1 | 1 | 1 |
| 42 | ¿En qué momento se ve a un caballo gris sacudir la cabeza con energía? | 217-223 | 1 | 1 | 4 | - |
| 43 | ¿Cuándo se muestran dos nutrias marinas flotando juntas boca arriba en el agua? | 223-227 | 1 | 1 | 1 | 1 |
| 44 | ¿En qué segundos aparece una ardilla gris comiendo una semilla sostenida con sus manos? | 227-232 | 1 | 1 | 2 | 1 |
| 45 | ¿En qué parte del vídeo aparece una tortuga marina nadando en un arrecife de coral? | 232-241 | 1 | 1 | 1 | 1 |
| 46 | ¿En qué momento aparece una esponja de mar amarilla porosa sobre fondo negro? | 241-243 | 1 | 1 | 1 | 1 |

---

## Apéndice B: Comparativa por consulta literal en `python_tutorial.mp4` (56 consultas)

Comparativa directa entre [`python_e2e_report.csv`](python_e2e_report.csv) (`Visual`, `2026-11-06`) y [`python_e2e_report_20261109.csv`](python_e2e_report_20261109.csv) (`Audio+Vis`, `2026-11-09`). El guion (`-`) indica ausencia de acierto en el top-5.

| # | Consulta | Intervalo (s) | Recup. Visual | Rerank Visual | Recup. Audio+Vis | Rerank Audio+Vis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | ¿En qué segundo aparece el presentador en blanco y negro saludando con las dos manos levantadas? | 0,0-0,7 | 3 | 1 | 4 | - |
| 2 | ¿Cuándo pasa el vídeo a color mostrando al presentador con el pulgar arriba junto al logotipo de la NASA? | 0,6-5,0 | 2 | 1 | 2 | 1 |
| 3 | ¿En qué momento aparecen en pantalla los logotipos de NASA, Google, Meta y Microsoft? | 2,5-9,3 | 1 | 1 | 1 | 1 |
| 4 | ¿En qué intervalo introduce el presentador el curso hablando a cámara antes de empezar los tipos de datos? | 9,3-19,0 | - | - | - | - |
| 5 | ¿Cuándo aparece el título 'Tipos de datos' en letras blancas grandes sobre el presentador? | 19,0-25,0 | 1 | 1 | 1 | 1 |
| 6 | ¿En qué segundos se muestra el rótulo 'int' para explicar los números enteros en Python? | 25,0-27,9 | 2 | 1 | **1** | 1 |
| 7 | ¿En qué momento aparece el rótulo 'float' en pantalla para explicar los números decimales? | 27,9-30,5 | 1 | 1 | 1 | 1 |
| 8 | ¿Cuándo se presenta el tipo de cadenas de texto mostrando únicamente el rótulo 'str'? | 30,5-33,5 | 3 | 2 | **1** | **1** |
| 9 | ¿En qué intervalo aparece el ejemplo de cadena de texto 'str' con ''Hola, Python'' en pantalla? | 33,5-37,0 | 1 | 1 | 1 | 1 |
| 10 | ¿En qué segundos aparece el rótulo 'bool' para introducir los valores booleanos? | 37,0-40,0 | 2 | 2 | 2 | 2 |
| 11 | ¿Cuándo se muestra en pantalla el tipo 'bool' acompañado del valor 'False'? | 40,0-44,8 | 1 | 1 | 1 | 1 |
| 12 | ¿En qué momento aparece el rótulo 'None' en pantalla para representar la ausencia de valor? | 44,8-47,5 | 1 | 1 | 1 | 1 |
| 13 | ¿En qué segundos se introduce el tipo de números complejos con el rótulo 'complex'? | 47,5-49,8 | 5 | 2 | **2** | 2 |
| 14 | ¿Cuándo aparece el ejemplo de número complejo 'complex' con la expresión '2 + 3j'? | 49,8-51,4 | 1 | 1 | 1 | 1 |
| 15 | ¿En qué intervalo cambia la imagen a blanco y negro con el presentador levantando la palma de la mano derecha? | 51,4-53,5 | 2 | 1 | 2 | 1 |
| 16 | ¿En qué momento aparece el título de sección 'Estructuras' en letras blancas grandes? | 53,5-57,5 | 1 | 1 | 1 | 1 |
| 17 | ¿Cuándo se explica la estructura 'list' con el ejemplo '['Brais', 37, True]' entre corchetes? | 57,5-66,1 | 1 | 1 | 1 | 1 |
| 18 | ¿En qué segundos levanta el presentador el puño cerrado con el rótulo 'tuple' en pantalla? | 66,1-71,1 | 1 | 1 | 1 | 1 |
| 19 | ¿En qué intervalo se muestra el ejemplo de tupla 'tuple' con '('Brais', 37, True)' entre paréntesis? | 71,1-76,1 | 1 | 1 | 1 | 1 |
| 20 | ¿Cuándo aparece la estructura 'set' con el ejemplo '{'Brais', 37, True, 37}' entre llaves? | 76,1-84,4 | 1 | 1 | 1 | 1 |
| 21 | ¿En qué segundos se explica el diccionario 'dict' con el ejemplo clave-valor '{'name': 'Brais'}'? | 84,4-96,1 | 1 | 1 | 1 | 1 |
| 22 | ¿En qué momento aparece el título 'Variables' en letras blancas grandes sobre el presentador? | 96,1-100,2 | 2 | 2 | **1** | **1** |
| 23 | ¿Cuándo se muestra por primera vez el editor Visual Studio Code con 'my_variable = 'Hola Python!'' en la línea 1? | 100,2-107,7 | 1 | 1 | 1 | 1 |
| 24 | ¿En qué segundos aparece el plano en blanco y negro del presentador haciendo el gesto del pulgar hacia arriba? | 109,0-110,2 | 1 | 1 | 1 | 1 |
| 25 | ¿En qué intervalo aparece la palabra 'Constantes' tachada con una cruz roja 'X'? | 110,2-114,1 | 1 | 1 | 1 | 1 |
| 26 | ¿Cuándo se ve en el editor la reasignación de 'my_variable' al número 6 y a 'true' con el menú de autocompletado? | 114,1-122,3 | 1 | 1 | 1 | 1 |
| 27 | ¿En qué segundos hace el presentador un gesto de precisión con la mano izquierda tras mostrar la reasignación de variables? | 122,3-127,3 | - | - | **1** | - |
| 28 | ¿En qué momento aparece el rótulo 'snake_case' en pantalla para explicar cómo nombrar variables? | 127,3-132,8 | 2 | 1 | **1** | 1 |
| 29 | ¿Cuándo se escribe 'my_snake_case_variable =' en la línea 1 del editor Visual Studio Code? | 132,8-139,2 | 1 | 1 | 1 | 1 |
| 30 | ¿En qué intervalo levanta el presentador el dedo índice izquierdo mostrando el tatuaje de su antebrazo? | 139,2-145,7 | 1 | 1 | 1 | 1 |
| 31 | ¿Cuándo aparece el título 'Funciones por defecto' sobreimpreso en dos líneas blancas? | 145,7-149,6 | 1 | 1 | 1 | 1 |
| 32 | ¿En qué segundos se muestra en el editor la función 'input' con la pregunta '¿Cuál es tu nombre?'? | 149,6-153,1 | 1 | 1 | 1 | 1 |
| 33 | ¿En qué momento se utiliza la función 'len(my_list)' en el editor para obtener la longitud de una lista? | 153,1-156,7 | 1 | 1 | 1 | 1 |
| 34 | ¿Cuándo aparece en el editor la instrucción 'print('Hola, Python!')' con el rótulo 'print' en grande? | 156,7-158,7 | - | 1 | **1** | 1 |
| 35 | ¿En qué segundos aparece el título de sección 'Ejecución' sobre el presentador? | 158,7-161,6 | 1 | 1 | 1 | 1 |
| 36 | ¿En qué intervalo se ejecuta el comando 'python3 hello_python.py' en la terminal imprimiendo 'Hola, Python!'? | 161,6-166,5 | 2 | 1 | **1** | 1 |
| 37 | ¿Cuándo se muestra la extensión 'Code Runner' de Jun Han en el Marketplace de Visual Studio Code? | 166,5-169,4 | 1 | 1 | 1 | 1 |
| 38 | ¿En qué segundos se abre el menú desplegable del botón de ejecución en VS Code con la opción 'Run Code' seleccionada? | 169,4-172,1 | 1 | 1 | 1 | 1 |
| 39 | ¿En qué momento aparece el presentador con gafas de sol pixeladas y lluvia de código binario verde al estilo Matrix? | 172,1-173,7 | 1 | 1 | 1 | 1 |
| 40 | ¿Cuándo aparece el título de sección 'Condicionales' sobre el presentador? | 173,7-175,6 | 1 | 1 | 1 | 1 |
| 41 | ¿En qué intervalo habla el presentador a cámara con el rótulo 'if' en grande en el centro? | 175,6-185,0 | 1 | 1 | 1 | 1 |
| 42 | ¿En qué segundos se escribe en el editor 'my_age = 37' y la cabecera 'if my_age >= 18:' sin el print todavía? | 185,0-191,0 | 4 | 1 | **1** | 1 |
| 43 | ¿Cuándo aparece en el editor la línea indentada 'print('Soy mayor de edad')' debajo de 'if my_age >= 18:'? | 191,0-197,6 | 1 | 1 | 1 | 1 |
| 44 | ¿En qué segundos enmarca el presentador con las manos el símbolo de llaves '{ }' aún sin tachar? | 197,6-201,8 | 1 | - | 1 | **1** |
| 45 | ¿En qué momento aparece el símbolo de llaves '{ }' tachado con una cruz roja 'X'? | 201,8-205,4 | - | - | - | - |
| 46 | ¿En qué intervalo se muestran las ramas 'elif my_age == 17' y 'else:' con 'print('Soy menor de edad')' en el editor? | 205,4-213,6 | 1 | 1 | 1 | 1 |
| 47 | ¿Cuándo aparece el título de sección 'Bucles' en letras blancas grandes sobre el presentador? | 213,6-221,1 | 1 | 1 | 1 | 1 |
| 48 | ¿En qué segundos se muestra en el editor el bucle 'for index in range(5):' con 'print(index)'? | 221,1-224,4 | 1 | 1 | 1 | 1 |
| 49 | ¿En qué segundo muestra el panel TERMINAL de VS Code la primera salida con los números del 0 al 4 tras ejecutar 'range(5)'? | 224,4-225,3 | - | - | - | **1** |
| 50 | ¿Cuándo se muestra en el editor el bucle 'for item in my_list:' recorriendo la lista '[1, 2, 3, 4, 5]'? | 225,3-228,1 | 1 | 1 | 1 | 1 |
| 51 | ¿En qué segundos muestra la terminal integrada de VS Code los números del 1 al 5 impresos tras recorrer 'my_list'? | 228,1-230,2 | 2 | 1 | **1** | 1 |
| 52 | ¿Cuándo aparece en el editor 'while index < 5' con el rótulo 'while' sobreimpreso en letras blancas? | 230,2-232,5 | 2 | 2 | **1** | **1** |
| 53 | ¿En qué intervalo se muestra el bucle 'while index < 5:' completo con el incremento 'index +=1' en la línea 5? | 232,5-236,0 | 1 | 1 | 1 | 1 |
| 54 | ¿En qué segundos muestra la terminal de VS Code la salida del 0 al 4 resultante de ejecutar el bucle 'while'? | 236,0-238,3 | 1 | - | 1 | **1** |
| 55 | ¿Cuándo aparece el título de sección 'Excepciones' en letras blancas grandes sobre el presentador? | 238,3-240,3 | 1 | 1 | 1 | 1 |
| 56 | ¿En qué segundos se muestra en el editor el bloque 'try / except' con la división por cero 'print(5 / 0)'? | 240,3-242,5 | 1 | 1 | 1 | 1 |
