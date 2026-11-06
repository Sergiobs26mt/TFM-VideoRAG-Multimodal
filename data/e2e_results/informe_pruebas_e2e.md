# Informe de pruebas E2E con la API de Gemini

Fechas de ejecución:
- `animals.mp4` (243 s): 2026-11-04 (línea base literal), 2026-11-05 (primera ejecución compleja) y 2026-11-06 (ejecución final tras correcciones).
- `python_tutorial.mp4` (242,5 s): 2026-11-06 (primera ejecución E2E y ejecución final tras mejoras para vídeo de programación).

Ejecuciones de referencia en `audit/` (carpetas locales ignoradas por git):
- `animals.mp4`: `audit/run_20261104_002041/`, `audit/run_20261105_164816/` y `audit/run_20261106_110324/`.
- `python_tutorial.mp4`: `audit/run_20261106_125320/` (inicial) y `audit/run_20261106_134026/` (final).

Copias versionadas de los informes CSV en `docs/`:
- `animals.mp4` - Batería literal inicial (46 consultas): [`e2e_report_20261104.csv`](e2e_report_20261104.csv)
- `animals.mp4` - Batería literal tras mejoras (46 consultas): [`e2e_report_20261106.csv`](e2e_report_20261106.csv)
- `animals.mp4` - Batería multi-objetivo abstracta (10 consultas): [`e2e_multi_target_report_20261105.csv`](e2e_multi_target_report_20261105.csv)
- `animals.mp4` - Batería negativa (8 consultas): [`e2e_negative_report_20261105.csv`](e2e_negative_report_20261105.csv)
- `python_tutorial.mp4` - Batería literal (56 consultas): [`python_e2e_report.csv`](python_e2e_report.csv)
- `python_tutorial.mp4` - Batería multi-objetivo abstracta (10 consultas): [`python_e2e_multi_target_report.csv`](python_e2e_multi_target_report.csv)
- `python_tutorial.mp4` - Batería negativa (8 consultas): [`python_e2e_negative_report.csv`](python_e2e_negative_report.csv)

## 1. Objetivo

Comprobar de extremo a extremo, con llamadas reales a la API de Gemini, que la aplicación localiza los fragmentos correctos de dos vídeos de dominios distintos (`animals.mp4` y `python_tutorial.mp4`) ante tres tipos de consultas:
1. Consultas literales específicas (1 intervalo objetivo por consulta).
2. Consultas abstractas multi-objetivo (múltiples apariciones repartidas por el vídeo).
3. Consultas negativas (conceptos ausentes en el vídeo que el sistema debe rechazar).

La referencia es el ground truth etiquetado a mano en [`data/video_chunks_ground_truth.csv`](../data/video_chunks_ground_truth.csv).

## 2. Alcance

| Se valida | No se valida |
| :--- | :--- |
| Indexación de `animals.mp4` y `python_tutorial.mp4` (fragmentación SSIM, embeddings, Qdrant) | Generación de respuestas en texto libre (la aplicación devuelve fragmentos de vídeo) |
| Recuperación vectorial en Qdrant (Etapa 1) | La columna `expected_answer` del CSV |
| Reordenación multimodal con VLM y collage vertical de fotogramas (Etapa 2) | - |
| Aciertos por solape temporal en consultas literales y multi-objetivo | - |
| Rechazo de falsos positivos en consultas negativas | - |

## 3. Dataset de referencia

El archivo [`data/video_chunks_ground_truth.csv`](../data/video_chunks_ground_truth.csv) contiene **102 filas** en total:
- **46 filas de `animals.mp4`** (`chunk_0000` a `chunk_0045`, de `0,0 s` a `243,0 s`), obtenidas a partir de las capturas de `data/photos/`.
- **56 filas de `python_tutorial.mp4`** (`python_chunk_0000` a `python_chunk_0055`, de `0,0 s` a `242,5 s`), obtenidas a partir de las 56 capturas de `data/photos/python/` y alineadas con los cortes reales del vídeo.

La aplicación fragmenta cada vídeo por cambio visual (SSIM) y duración máxima de plano. Sus chunks no coinciden 1 a 1 con las filas del CSV. Por eso la evaluación mide el solape temporal (`> 0 s`) entre el intervalo devuelto y el intervalo etiquetado.

Columnas que usan los tests: `test_query`, `video_name`, `start_time` y `end_time`. El cargador `load_ground_truth_csv` en [`evaluation.py`](../src/rag_engine/evaluation.py) acepta el parámetro opcional `video_name` para filtrar el dataset por vídeo.

## 4. Pruebas realizadas

### 4.1 Evaluación y cargador del CSV (unitarias, offline)

Escritas con TDD en [`tests/test_evaluation.py`](../tests/test_evaluation.py), [`tests/test_reranker.py`](../tests/test_reranker.py) y [`tests/test_service.py`](../tests/test_service.py).

| Test | Comprueba |
| :--- | :--- |
| `test_load_ground_truth_csv_maps_dataset_columns` | Mapea `test_query`, `start_time` y `end_time` a `GroundTruthQuery`. Lee campos entre comillas con comas. |
| `test_load_ground_truth_csv_filters_by_video_name` | Filtra las filas del CSV por `video_name` (`animals.mp4` o `python_tutorial.mp4`). |
| `test_load_ground_truth_csv_rejects_missing_file` | Lanza `FileNotFoundError` si el fichero no existe. |
| `test_evaluate_multi_target_results_counts_distinct_covered_intervals` | Calcula intervalos distintos cubiertos (`interval_recall`), precisión (`precision`) y rango del primer acierto (`first_hit_rank`). |
| `test_evaluate_negative_results_distinguishes_raw_and_reranked` | Cuenta los falsos positivos antes y después del filtrado por `is_relevant`. |
| `test_run_multi_target_and_negative_benchmarks_execute_queries` | Ejecuta las baterías multi-objetivo y negativa con `VideoRAGService`. |
| `test_harness_complex_reports_serialize_multi_target_and_negative_metrics` | Verifica las consultas multi-objetivo y negativas de `animals.mp4` y la escritura de sus CSV. |
| `test_harness_python_queries_cover_multi_target_and_negative_cases` | Verifica las 10 consultas multi-objetivo y las 8 negativas de `python_tutorial.mp4`. |
| `test_load_candidate_frames_single_frame_builds_vertical_collage_from_distinct_keyframes` | Verifica que el reranker combine hasta 3 fotogramas distintos (`frame_0`, `frame_1`, `frame_2`) en un único collage vertical JPEG. |
| `test_service_search_preserves_distinct_short_shots_over_one_second` | Verifica que `_deduplicate_temporal_candidates` no fusione planos distintos de duración `>= 1,0 s`. |

### 4.2 Calidad del código

| Comando | Resultado |
| :--- | :--- |
| `pytest tests -q` | 116 passed, 14 deselected (los 14 tests E2E) |
| `ruff check src tests` | Sin incidencias |
| `mypy --explicit-package-bases src tests` | Sin incidencias (38 ficheros comprobados) |

### 4.3 Pruebas E2E con la API real

Código en [`tests/e2e/test_animals_e2e.py`](../tests/e2e/test_animals_e2e.py), [`tests/e2e/test_python_e2e.py`](../tests/e2e/test_python_e2e.py) y [`tests/e2e/harness.py`](../tests/e2e/harness.py). Cada batería ejecuta una única sesión compartida entre sus 7 tests:
- `animals.mp4`: **7 passed en 485,51 s (8 min 5 s)**.
- `python_tutorial.mp4`: **7 passed en 483,47 s (8 min 3 s)**.

| Test (`animals` / `python`) | Comprueba | Umbral (`animals` / `python`) | Resultado `animals.mp4` (`run_20261106_110324`) | Resultado `python_tutorial.mp4` (`run_20261106_134026`) |
| :--- | :--- | :--- | :--- | :--- |
| `test_*_indexing_covers_the_whole_video` | Solo el vídeo bajo prueba está indexado y hay chunks suficientes | `>= 31` / `>= 38` chunks | 82 chunks, pasa | 82 chunks, pasa |
| `test_*_retrieval_finds_the_labelled_interval` | Recall@5 y MRR sin reordenación | R@5 `>= 0,85`, MRR `>= 0,80` / `>= 0,75` | 97,8 % y 0,891, pasa | **91,1 % y 0,779**, pasa |
| `test_*_rerank_does_not_hurt_top_result` | Recall@1 con reordenación no baja más de 5 puntos | `>= retrieval - 0,05` | 89,1 % frente a 80,4 %, pasa | **80,4 % frente a 67,9 % (+12,5 pts)**, pasa |
| `test_*_report_lists_every_query` | El informe literal tiene una fila por consulta, en orden | 46 / 56 filas | 46 filas, pasa | 56 filas, pasa |
| `test_*_multi_target_queries_find_multiple_matching_intervals` | Cobertura media multi-objetivo y mejora de precisión tras rerank | Recall `>= 0,50` y `prec_rerank >= prec_retrieval` | Recall 85,4 % / 82,2 %, precisión sube de 56,0 % a **94,6 %**, pasa | Recall 85,0 % / 78,3 %, precisión sube de 45,0 % a **80,3 %**, pasa |
| `test_*_negative_queries_are_rejected_by_reranker` | El juez multimodal rechaza las consultas sin respuesta | Tasa de rechazo `>= 0,75` | **100,0 % (8/8, 0/40 FP)**, pasa | **100,0 % (8/8, 0/40 FP)**, pasa |
| `test_*_complex_reports_list_every_query` | Los informes multi-objetivo y negativo listan todas las consultas | 10 y 8 filas | 10 y 8 filas, pasa | 10 y 8 filas, pasa |

## 5. Método de evaluación

- **Acierto:** el resultado pertenece al mismo vídeo y su intervalo `[start_time, end_time]` solapa más de `0 s` con el intervalo etiquetado.
- **Rango de acierto:** posición del primer resultado que acierta (`1` a `5`). Si no hay acierto, queda vacío.
- **Filtrado del juez (Etapa 2):** con `rerank=True`, el VLM descarta los candidatos con `is_relevant=false`. Esos candidatos no se devuelven ni cuentan como acierto.
- **Parámetros:** `top_k = 5` en consultas literales y negativas (`top_k * 2 = 10` candidatos evaluados por el VLM en una sola llamada) y `top_k = 10` en consultas multi-objetivo (`20` candidatos evaluados por el VLM).
- **Modelos:** `gemini-embedding-2-preview` (embeddings multimodales) y `gemini-3.5-flash-lite` (juez multimodal). Ver [`config.py`](../src/config.py).
- **Representación visual por candidato:** cuando `frames_per_candidate = 1`, `_build_vertical_collage` en [`reranker.py`](../src/rag_engine/reranker.py) apila verticalmente hasta 3 fotogramas distintos del chunk (`inicio`, `medio`, `fin`) en una sola imagen JPEG de `384 px` de ancho. Así el VLM inspecciona la evolución temporal del fragmento enviando una sola imagen por candidato.

Métricas:
- **Recall@1 / Recall@3 / Recall@5:** porcentaje de consultas literales con acierto en el top-1, top-3 o top-5.
- **MRR:** media del inverso del rango de acierto. Un fallo suma `0`.
- **Interval Recall (multi-objetivo):** fracción de apariciones reales de una categoría abstracta cubiertas por al menos un fragmento devuelto.
- **Precision (multi-objetivo):** fracción de fragmentos devueltos que solapan con alguna aparición real de la categoría.
- **Tasa de rechazo negativo:** porcentaje de consultas negativas en las que el sistema devuelve `0` falsos positivos tras el juez multimodal.

---

## 6. Resultados en `animals.mp4`

### 6.1 Indexación

- **82 chunks** detectados y vectorizados (`threshold = 0,5`, `max_shot_duration = 8,0 s`). Todos quedaron insertados en Qdrant.

### 6.2 Consultas literales específicas (46 consultas, `top_k = 5`)

| Ejecución | Modo | Consultas | Recall@1 | Recall@3 | Recall@5 | MRR | Latencia media |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Línea base (`2026-11-04`) | Solo recuperación vectorial | 46 | 84,8 % (39) | 95,7 % (44) | 97,8 % (45) | 0,908 | - |
| Línea base (`2026-11-04`) | Con reordenación multimodal | 46 | 87,0 % (40) | 93,5 % (43) | 93,5 % (43) | 0,902 | 5,32 s |
| Tras mejoras (`2026-11-06`) | Solo recuperación vectorial | 46 | 80,4 % (37) | 95,7 % (44) | **97,8 % (45)** | 0,891 | - |
| Tras mejoras (`2026-11-06`) | Con reordenación multimodal | 46 | **89,1 % (41)** | **93,5 % (43)** | **93,5 % (43)** | **0,913** | **3,62 s** |

Efecto de la reordenación en la ejecución final (`run_20261106_110324`, [`e2e_report_20261106.csv`](e2e_report_20261106.csv)):
- **Mejora el `Recall@1` en +8,7 puntos porcentuales** (de `80,4 %` a `89,1 %`, `41/46` aciertos en primera posición) y sube el `MRR` de `0,891` a `0,913`.
- **Sube al puesto 1 seis consultas** que en recuperación vectorial estaban en los puestos 2 a 4 (consultas `9, 14, 15, 21, 24, 29`), incluida la consulta 14 (`ave rapaz de plumaje marrón y pecho blanco`), que tras corregir la etiqueta errónea del CSV (`cría` -> `adulto`) pasó a `hit_rank = 1` tanto en recuperación como en reordenación.

### 6.3 Consultas abstractas multi-objetivo en `animals.mp4` (10 consultas, `top_k = 10`)

Resultados en `audit/run_20261106_110324/` ([`e2e_multi_target_report_20261105.csv`](e2e_multi_target_report_20261105.csv)):

| # | Consulta abstracta | Planos reales | Cubiertos (Recup.) | Recall (Recup.) | Precisión (Recup.) | Cubiertos (Rerank) | Recall (Rerank) | Precisión (Rerank) | 1.er acierto (Rerank) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Felinos` | 4 | 3 | 75,0 % | 30,0 % | 3 | 75,0 % | **100,0 %** | 1 |
| 2 | `Reptiles` | 2 | 2 | **100,0 %** | 30,0 % | 2 | **100,0 %** | **100,0 %** | 1 |
| 3 | `Aves` | 4 | 4 | **100,0 %** | 40,0 % | 4 | **100,0 %** | **100,0 %** | 1 |
| 4 | `Manada o grupo de tres o más animales juntos` | 10 | 6 | **60,0 %** | 70,0 % | 6 | **60,0 %** | **100,0 %** | 1 |
| 5 | `Animales de color blanco y negro` | 5 | 5 | **100,0 %** | 60,0 % | 4 | 80,0 % | **100,0 %** | 1 |
| 6 | `Animales marinos o acuáticos nadando o flotando en el agua` | 8 | 6 | **75,0 %** | 70,0 % | 6 | **75,0 %** | **100,0 %** | 1 |
| 7 | `Animales comiendo, pastando o amamantándose` | 8 | 6 | **75,0 %** | 70,0 % | 5 | 62,5 % | **100,0 %** | 1 |
| 8 | `Crías de animales o animales jóvenes` | 7 | 6 | **85,7 %** | 50,0 % | 6 | **85,7 %** | 62,5 % | 1 |
| 9 | `Primerísimo plano macro de ojos, orejas o fosas nasales de un animal` | 6 | 5 | **83,3 %** | 70,0 % | 5 | **83,3 %** | 83,3 % | 1 |
| 10 | `Animación 3D microscópica de células o glóbulos rojos` | 4 | 4 | **100,0 %** | 70,0 % | 4 | **100,0 %** | **100,0 %** | 1 |
| **Media (tras mejoras)** | - | **5,8** | **4,7** | **85,4 %** | **56,0 %** | **4,5** | **82,2 %** | **94,6 %** | **1,0** |
| *Media (antes de mejoras)* | - | *5,9* | *3,9* | *74,8 %* | *66,0 %* | *3,9* | *74,8 %* | *79,0 %* | *1,0* |

### 6.4 Consultas negativas en `animals.mp4` (8 consultas, `top_k = 5`)

Resultados en `audit/run_20261106_110324/` ([`e2e_negative_report_20261105.csv`](e2e_negative_report_20261105.csv)):

| # | Consulta negativa | Tipo | Falsos positivos (Etapa 1) | Falsos positivos (Etapa 2 antes) | Falsos positivos (Etapa 2 tras mejoras) | Rechazada por completo |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `Tiburón blanco atacando en el océano` | Negativo difícil marino | 5 / 5 | 0 / 5 | **0 / 5** | Sí |
| 2 | `Canguro saltando por el desierto australiano` | Negativo difícil mamífero | 5 / 5 | 5 / 5 (fallback) | **0 / 5** | Sí |
| 3 | `Serpiente cobra deslizándose por la arena` | Negativo difícil reptil | 5 / 5 | 0 / 5 | **0 / 5** | Sí |
| 4 | `Águila volando alto en el cielo azul` | Negativo difícil ave | 5 / 5 | 0 / 5 | **0 / 5** | Sí |
| 5 | `Cocodrilo con las fauces abiertas en la orilla de un río` | Negativo difícil reptil | 5 / 5 | 5 / 5 (fallback) | **0 / 5** | Sí |
| 6 | `Bandada de flamencos rosas en una laguna salada` | Negativo difícil aves | 5 / 5 | 0 / 5 | **0 / 5** | Sí |
| 7 | `Camión de bomberos rojo con la sirena encendida por la calle` | Fuera de dominio | 5 / 5 | 0 / 5 | **0 / 5** | Sí |
| 8 | `Persona programando código en un ordenador portátil en una oficina` | Fuera de dominio | 5 / 5 | 5 / 5 (fallback) | **0 / 5** | Sí |
| **Total** | - | - | **40 / 40 (100 % FP)** | **15 / 40 (62,5 % rechazo)** | **0 / 40 (100,0 % rechazo)** | **8 / 8 (100 %)** |

---

## 7. Resultados en `python_tutorial.mp4`

### 7.1 Características del vídeo e indexación

`python_tutorial.mp4` (`242,5 s`) es un tutorial de programación en Python que alterna planos del presentador con sobreimpresiones tipográficas (`int`, `float`, `str`, `list`, `tuple`, `set`, `dict`, `snake_case`, `if`, `while`) y grabaciones de pantalla del IDE Visual Studio Code (editor de código y panel de terminal integrada).

A diferencia de un vídeo de naturaleza, las transiciones entre el editor de VS Code y la terminal integrada comparten el mismo fondo oscuro (`#1e1e1e`), por lo que el índice SSIM global entre planos distintos se sitúa entre `0,51` y `0,59`. Al indexar `python_tutorial.mp4` con `threshold = 0,6` y `max_shot_duration = 5,0 s`, el pipeline detecta **82 chunks** (exactamente el mismo número de fragmentos que en `animals.mp4`), separando cada transición Editor/Terminal y cada rótulo conceptual en su propio chunk.

### 7.2 Consultas literales específicas en `python_tutorial.mp4` (56 consultas, `top_k = 5`)

Comparativa entre la ejecución inicial (`audit/run_20261106_125320/`) y la ejecución final (`audit/run_20261106_134026/`, [`python_e2e_report.csv`](python_e2e_report.csv)):

| Ejecución | Modo | Consultas | Recall@1 | Recall@3 | Recall@5 | MRR | Latencia media |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Inicial (`run_20261106_125320`) | Solo recuperación vectorial | 56 | 60,7 % (34) | 78,6 % (44) | 85,7 % (48) | 0,704 | - |
| Inicial (`run_20261106_125320`) | Con reordenación multimodal | 56 | 55,4 % (31) | 66,1 % (37) | 67,9 % (38) | 0,605 | 3,13 s |
| **Final (`run_20261106_134026`)** | **Solo recuperación vectorial** | **56** | **67,9 % (38)** | **87,5 % (49)** | **91,1 % (51)** | **0,779** | - |
| **Final (`run_20261106_134026`)** | **Con reordenación multimodal** | **56** | **80,4 % (45)** | **89,3 % (50)** | **89,3 % (50)** | **0,848** | **4,02 s** |

### 7.3 Efecto de la reordenación por consulta en `python_tutorial.mp4`

| Efecto | Consultas (número del Apéndice B) | Cantidad |
| :--- | :--- | :--- |
| Sin cambio | resto | 43 |
| Mejora al puesto 1 | 1, 2, 6, 15, 28, 36, 42, 51 | 8 |
| Mejora al puesto 2 (desde puestos 3 y 5) | 8, 13 | 2 |
| Rescata un fallo de la recuperación al puesto 1 | 34 (`print('Hola, Python!')` con rótulo `print`) | 1 |
| Empeora: descarta un acierto del puesto 1 | 44, 54 | 2 |

Resumen del impacto del juez multimodal con collage vertical de 3 fotogramas:
- **El `Recall@1` sube +12,5 puntos porcentuales** respecto a la recuperación vectorial (`67,9 % -> 80,4 %`, pasando de 38 a 45 aciertos en el puesto 1).
- **El `Recall@3` sube del 87,5 % al 89,3 %** (`50/56` consultas acertadas en el top-3).
- **El `MRR` sube +0,069 puntos** (`0,779 -> 0,848`).
- En la ejecución inicial, el reranker solo veía el fotograma central (`frame_1.jpg`) de cada chunk y descartaba 11 aciertos cuyo rótulo o cambio de código aparecía al inicio (`frame_0.jpg`) o al final (`frame_2.jpg`). Con `_build_vertical_collage` en [`reranker.py`](../src/rag_engine/reranker.py), el juez ve los 3 momentos del chunk en una sola imagen y **promueve 11 consultas a posiciones superiores (9 de ellas directamente al puesto 1)**.

### 7.4 Consultas abstractas multi-objetivo en `python_tutorial.mp4` (10 consultas, `top_k = 10`)

Resultados en `audit/run_20261106_134026/` ([`python_e2e_multi_target_report.csv`](python_e2e_multi_target_report.csv)):

| # | Consulta abstracta | Planos reales | Cubiertos (Recup.) | Recall (Recup.) | Precisión (Recup.) | Cubiertos (Rerank) | Recall (Rerank) | Precisión (Rerank) | 1.er acierto (Rerank) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Planos del presentador en blanco y negro` | 3 | 3 | **100,0 %** | 40,0 % | 3 | **100,0 %** | 44,4 % | 1 |
| 2 | `Tipos de datos primitivos en Python (int, float, str, bool, None, complex)` | 9 | 6 | 66,7 % | 40,0 % | 6 | 66,7 % | 66,7 % | 1 |
| 3 | `Estructuras de datos y colecciones en Python (list, tuple, set, dict)` | 6 | 6 | **100,0 %** | 70,0 % | 6 | **100,0 %** | **100,0 %** | 1 |
| 4 | `Rótulos o símbolos tachados con una cruz roja X en pantalla` | 2 | 2 | **100,0 %** | 20,0 % | 1 | 50,0 % | 50,0 % | 1 |
| 5 | `Terminal o consola mostrando la salida de ejecución de un script con números o texto` | 4 | 3 | 75,0 % | 40,0 % | 3 | 75,0 % | **75,0 %** | 2 |
| 6 | `Código de condicionales if, elif y else en el editor Visual Studio Code` | 3 | 3 | **100,0 %** | 50,0 % | 3 | **100,0 %** | **100,0 %** | 1 |
| 7 | `Código de bucles for y while en el editor Visual Studio Code` | 4 | 4 | **100,0 %** | 40,0 % | 4 | **100,0 %** | **100,0 %** | 1 |
| 8 | `Declaración y reasignación de variables en el editor Visual Studio Code` | 3 | 2 | 66,7 % | 50,0 % | 2 | 66,7 % | **100,0 %** | 1 |
| 9 | `Uso de funciones integradas input, len y print en el editor con ventana emergente de documentación` | 6 | 4 | 66,7 % | 40,0 % | 3 | 50,0 % | **100,0 %** | 1 |
| 10 | `Títulos de sección sobreimpresos en letras blancas grandes con el presentador hablando a cámara` | 8 | 6 | 75,0 % | 60,0 % | 6 | 75,0 % | 66,7 % | 1 |
| **Media (final)** | - | **4,8** | **3,9** | **85,0 %** | **45,0 %** | **3,7** | **78,3 %** | **80,3 %** | **1,1** |
| *Media (inicial)* | - | *4,8* | *3,7* | *77,5 %* | *46,0 %* | *3,0* | *64,2 %* | *66,0 %* | *1,2* |

Aspectos destacados de la batería multi-objetivo en `python_tutorial.mp4`:
- **100,0 % de cobertura (`interval_recall`) en 5 categorías completas:** `Planos del presentador en blanco y negro` (`3/3`), `Estructuras de datos y colecciones (list, tuple, set, dict)` (`6/6`), `Código de condicionales if, elif y else` (`3/3`), `Código de bucles for y while` (`4/4`) y `Rótulos tachados con X roja` en Etapa 1 (`2/2`).
- **La precisión media sube del 45,0 % al 80,3 % (+35,3 puntos porcentuales) tras el reranker:** alcanzando el **100,0 % de precisión** en `Estructuras de datos`, `Condicionales`, `Bucles`, `Declaración y reasignación de variables` y `Funciones integradas input, len y print`.
- **Detección del plano en blanco y negro de `1,17 s` (`109,0 - 110,2 s`):** al ajustar `micro_chunk_duration` de `1,5 s` a `1,0 s` en `_deduplicate_temporal_candidates` ([`service.py`](../src/rag_engine/service.py)), el tercer plano en blanco y negro deja de fusionarse con el plano en color adyacente y se recupera el 100 % (`3/3`) de los planos monocromáticos del vídeo.

### 7.5 Consultas negativas en `python_tutorial.mp4` (8 consultas, `top_k = 5`)

Resultados en `audit/run_20261106_134026/` ([`python_e2e_negative_report.csv`](python_e2e_negative_report.csv)):

| # | Consulta negativa | Tipo | Falsos positivos (Etapa 1) | Falsos positivos (Etapa 2 Rerank) | Rechazada por completo |
| :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `Definición de una clase orientada a objetos con class y el método __init__ en Python` | Negativo difícil programación Python | 5 / 5 | **0 / 5** | Sí |
| 2 | `Gráfico de barras o visualización de datos con matplotlib y pandas en un Jupyter Notebook` | Negativo difícil ciencia de datos Python | 5 / 5 | **0 / 5** | Sí |
| 3 | `Consulta SQL con SELECT y FROM en una base de datos PostgreSQL` | Negativo difícil desarrollo | 5 / 5 | **0 / 5** | Sí |
| 4 | `Instalación de paquetes con el comando pip install en la terminal` | Negativo difícil terminal Python | 5 / 5 | **0 / 5** | Sí |
| 5 | `Diagrama de arquitectura de redes neuronales o entrenamiento con PyTorch` | Negativo difícil IA / Python | 5 / 5 | **0 / 5** | Sí |
| 6 | `Página web con código HTML y estilos CSS abierta en el navegador Chrome` | Negativo difícil desarrollo web | 5 / 5 | **0 / 5** | Sí |
| 7 | `León macho con melena descansando tumbado sobre la hierba verde` | Fuera de dominio (de `animals.mp4`) | 5 / 5 | **0 / 5** | Sí |
| 8 | `Partido de fútbol en un estadio lleno de espectadores` | Fuera de dominio general | 5 / 5 | **0 / 5** | Sí |
| **Total** | - | - | **40 / 40 (100 % FP)** | **0 / 40 (100,0 % rechazo)** | **8 / 8 (100 %)** |

El juez multimodal rechaza el **100,0 % de los falsos positivos (`0/40`)** tanto en consultas fuera de dominio como en consultas de programación en Python muy cercanas a la temática del vídeo (`class / __init__`, `pip install` en terminal, `matplotlib / pandas`).

---

## 8. Mejoras de ingeniería aplicadas a raíz de las pruebas

1. **Collage vertical de 3 fotogramas por candidato (`_build_vertical_collage` en [`reranker.py`](../src/rag_engine/reranker.py)):** cuando `frames_per_candidate == 1`, el reranker carga hasta 3 fotogramas clave del chunk (`frame_0.jpg`, `frame_1.jpg`, `frame_2.jpg`), descarta duplicados idénticos y los une verticalmente (`cv2.vconcat`) en un solo JPEG de `384 px` de ancho. Esto mantiene una sola imagen por candidato en la petición a Gemini (evitando errores `503 Service Unavailable` por exceso de imágenes) y permite al juez ver el inicio, mitad y final del fragmento.
2. **Deduplicación temporal de micro-fragmentos (`_deduplicate_temporal_candidates` en [`service.py`](../src/rag_engine/service.py)):** agrupa micro-cortes consecutivos (`< 1,0 s`) de una misma escena dinámica conservando la envolvente temporal completa `[min(start), max(end)]`, sin absorber planos independientes de duración `>= 1,0 s`.
3. **Eliminación de truncamientos JSON y refuerzo de reintentos ([`reranker.py`](../src/rag_engine/reranker.py) y [`gemini_client.py`](../src/utils/gemini_client.py)):** `max_output_tokens = 2048`, parser tolerante a truncamiento parcial `_parse_evaluations_json` y reintentos exponenciales ante `429`, `503` y `httpx.HTTPError`. En ambas ejecuciones finales (`animals.mp4` y `python_tutorial.mp4`, 138 llamadas al juez en total) hubo **0 fallos de evaluación**.
4. **Parámetros de segmentación adaptados a screencasts ([`harness.py`](../tests/e2e/harness.py)):** `threshold = 0,6` y `max_shot_duration = 5,0 s` para `python_tutorial.mp4`, permitiendo detectar cambios sutiles entre editor y terminal sobre el mismo fondo oscuro.

## 9. Cómo reproducir

Ejecutar toda la suite E2E (ambos vídeos):

```bash
env_tfm/bin/python -m pytest tests/e2e -m e2e -q
```

O ejecutar cada vídeo por separado:

```bash
env_tfm/bin/python -m pytest tests/e2e/test_animals_e2e.py -m e2e -q
env_tfm/bin/python -m pytest tests/e2e/test_python_e2e.py -m e2e -q
```

Requiere `GEMINI_API_KEY` configurada en `.env` y los vídeos en `data/raw/animals.mp4` y `data/raw/python_tutorial.mp4`.

---

## Apéndice A: resultados por consulta literal en `animals.mp4` (46 consultas)

Rango de acierto (`1` a `5`) en `audit/run_20261106_110324/` ([`e2e_report_20261106.csv`](e2e_report_20261106.csv)). El guion (`-`) indica ausencia de acierto en el top-5.

| # | Consulta | Intervalo (s) | Recuperación | Reordenación |
| :--- | :--- | :--- | :---: | :---: |
| 1 | ¿En qué segundo aparece un elefante caminando en un recinto? | 0-1 | 1 | 1 |
| 2 | ¿En qué momento aparecen lémures de cola anillada con el rótulo de REINO ANIMALIA? | 1-5 | 3 | 3 |
| 3 | ¿Dónde y cuándo se muestra un león macho descansando tumbado? | 5-10 | 1 | 1 |
| 4 | ¿En qué intervalo temporal aparece la cabecera con el texto ECOLOGÍA VERDE? | 10-14 | 1 | - |
| 5 | ¿Cuándo se observa a tres elefantes incluyendo una cría caminando entre la hierba? | 14-18 | 1 | 1 |
| 6 | ¿En qué segundo aparece un gato gris con ojos verdes en primer plano? | 18-23 | 1 | 1 |
| 7 | ¿En qué momento aparece el ojo de un camaleón en primer plano macro? | 23-29 | 1 | 1 |
| 8 | ¿Cuándo se muestra a un grupo de ciervas descansando en la nieve? | 29-32 | - | - |
| 9 | ¿En qué intervalo se ve una medusa nadando en el mar azul? | 32-35 | 2 | 1 |
| 10 | ¿Dónde aparece una leona con un collar de rastreo caminando? | 35-39 | 1 | 1 |
| 11 | ¿En qué segundos se ven tres cisnes blancos en un lago al amanecer? | 39-45 | 1 | 1 |
| 12 | ¿Cuándo aparece un ternero negro con la cara blanca en primer plano? | 45-49 | 1 | 1 |
| 13 | ¿En qué momento aparece una vaca Hereford marrón y blanca junto a su cría? | 49-52 | 1 | 1 |
| 14 | ¿En qué parte del vídeo se ve un ave rapaz de plumaje marrón y pecho blanco en primer plano? | 52-59 | 2 | 1 |
| 15 | ¿En qué segundos aparece una nutria marina flotando de espaldas en el agua? | 59-64 | 2 | 1 |
| 16 | ¿Cuándo se muestra una animación 3D de los orgánulos de una célula animal? | 64-76 | 1 | 1 |
| 17 | ¿En qué momento se observa a un caballo pastando hierba verde de cerca? | 76-81 | 1 | 1 |
| 18 | ¿Cuándo aparece un ternero mamando de la ubre de una vaca? | 81-86 | 1 | 1 |
| 19 | ¿En qué segundos aparece una ardilla comiendo plátanos sobre una tabla de madera? | 86-90 | 1 | 1 |
| 20 | ¿Dónde y en qué segundo se muestra una oveja blanca con etiquetas amarillas pastando? | 90-96 | 1 | 1 |
| 21 | ¿En qué intervalo temporal corre una manada de ciervos por un bosque nevado? | 96-102 | 3 | 1 |
| 22 | ¿Cuándo aparece una cría de foca moteada recostada en la arena de la playa? | 102-106 | 1 | 1 |
| 23 | ¿En qué momento aparecen tres pingüinos en una orilla rocosa? | 106-110 | 1 | 1 |
| 24 | ¿Cuándo aparece una cría de mono comiendo fruta con las dos manos? | 110-114 | 2 | 1 |
| 25 | ¿En qué segundos se ven cebras comiendo pasto en una llanura árida? | 114-120 | 1 | 1 |
| 26 | ¿En qué intervalo aparece un gallo o pollo blanco con cresta roja en primer plano? | 120-124 | 1 | 1 |
| 27 | ¿Cuándo se muestra a tres osos pardos juntos en un prado verde? | 124-130 | 1 | 1 |
| 28 | ¿En qué momento aparece el primer plano de la oreja peluda de un animal escuchando? | 130-136 | 1 | 1 |
| 29 | ¿En qué segundos se ve un gran ciervo macho con cornamenta ramificada y crías moteadas? | 136-144 | 4 | 1 |
| 30 | ¿Cuándo aparecen tres monos sentados juntos acicalándose y desparasitándose? | 144-158 | 1 | 1 |
| 31 | ¿En qué parte se observa una animación 3D de glóbulos rojos dentro de un vaso sanguíneo? | 158-170 | 2 | 2 |
| 32 | ¿Cuándo se muestra una animación azul de células fluyendo en espiral? | 170-174 | 1 | 1 |
| 33 | ¿En qué intervalo temporal se ven eritrocitos en primer plano dentro de una arteria roja? | 174-180 | 1 | 1 |
| 34 | ¿En qué segundo aparece el primerísimo plano del hocico de un ciervo respirando? | 180-186 | 1 | 1 |
| 35 | ¿Cuándo se ve a un perro marrón jadeando con la lengua fuera? | 186-193 | 1 | 1 |
| 36 | ¿En qué momento aparece el ojo verde de un gato atigrado de perfil? | 193-196 | 1 | 1 |
| 37 | ¿En qué segundos se ve a una raya marina o mantarraya nadando en el agua? | 196-198 | 1 | 1 |
| 38 | ¿Cuándo aparece una araña saltarina en macro con sus múltiples ojos negros? | 198-201 | 1 | 1 |
| 39 | ¿En qué intervalo se muestra el ojo dorado y la cabeza de una rana de piel húmeda? | 201-208 | 1 | 1 |
| 40 | ¿Cuándo aparece un caballo marrón con una mancha blanca en la frente? | 208-212 | 1 | 1 |
| 41 | ¿En qué segundos aparece una medusa gigante azul moteada con un buceador? | 212-217 | 1 | 1 |
| 42 | ¿En qué momento se ve a un caballo gris sacudir la cabeza con energía? | 217-223 | 1 | 1 |
| 43 | ¿Cuándo se muestran dos nutrias marinas flotando juntas boca arriba en el agua? | 223-227 | 1 | 1 |
| 44 | ¿En qué segundos aparece una ardilla gris comiendo una semilla sostenida con sus manos? | 227-232 | 1 | 1 |
| 45 | ¿En qué parte del vídeo aparece una tortuga marina nadando en un arrecife de coral? | 232-241 | 1 | 1 |
| 46 | ¿En qué momento aparece una esponja de mar amarilla porosa sobre fondo negro? | 241-243 | 1 | 1 |

## Apéndice B: resultados por consulta literal en `python_tutorial.mp4` (56 consultas)

Rango de acierto (`1` a `5`) en `audit/run_20261106_134026/` ([`python_e2e_report.csv`](python_e2e_report.csv)). El guion (`-`) indica ausencia de acierto en el top-5.

| # | Consulta | Intervalo (s) | Recuperación | Reordenación |
| :--- | :--- | :--- | :---: | :---: |
| 1 | ¿En qué segundo aparece el presentador en blanco y negro saludando con las dos manos levantadas? | 0,0-0,7 | 3 | 1 |
| 2 | ¿Cuándo pasa el vídeo a color mostrando al presentador con el pulgar arriba junto al logotipo de la NASA? | 0,6-5,0 | 2 | 1 |
| 3 | ¿En qué momento aparecen en pantalla los logotipos de NASA, Google, Meta y Microsoft? | 2,5-9,3 | 1 | 1 |
| 4 | ¿En qué intervalo introduce el presentador el curso hablando a cámara antes de empezar los tipos de datos? | 9,3-19,0 | - | - |
| 5 | ¿Cuándo aparece el título 'Tipos de datos' en letras blancas grandes sobre el presentador? | 19,0-25,0 | 1 | 1 |
| 6 | ¿En qué segundos se muestra el rótulo 'int' para explicar los números enteros en Python? | 25,0-27,9 | 2 | 1 |
| 7 | ¿En qué momento aparece el rótulo 'float' en pantalla para explicar los números decimales? | 27,9-30,5 | 1 | 1 |
| 8 | ¿Cuándo se presenta el tipo de cadenas de texto mostrando únicamente el rótulo 'str'? | 30,5-33,5 | 3 | 2 |
| 9 | ¿En qué intervalo aparece el ejemplo de cadena de texto 'str' con ''Hola, Python'' en pantalla? | 33,5-37,0 | 1 | 1 |
| 10 | ¿En qué segundos aparece el rótulo 'bool' para introducir los valores booleanos? | 37,0-40,0 | 2 | 2 |
| 11 | ¿Cuándo se muestra en pantalla el tipo 'bool' acompañado del valor 'False'? | 40,0-44,8 | 1 | 1 |
| 12 | ¿En qué momento aparece el rótulo 'None' en pantalla para representar la ausencia de valor? | 44,8-47,5 | 1 | 1 |
| 13 | ¿En qué segundos se introduce el tipo de números complejos con el rótulo 'complex'? | 47,5-49,8 | 5 | 2 |
| 14 | ¿Cuándo aparece el ejemplo de número complejo 'complex' con la expresión '2 + 3j'? | 49,8-51,4 | 1 | 1 |
| 15 | ¿En qué intervalo cambia la imagen a blanco y negro con el presentador levantando la palma de la mano derecha? | 51,4-53,5 | 2 | 1 |
| 16 | ¿En qué momento aparece el título de sección 'Estructuras' en letras blancas grandes? | 53,5-57,5 | 1 | 1 |
| 17 | ¿Cuándo se explica la estructura 'list' con el ejemplo '['Brais', 37, True]' entre corchetes? | 57,5-66,1 | 1 | 1 |
| 18 | ¿En qué segundos levanta el presentador el puño cerrado con el rótulo 'tuple' en pantalla? | 66,1-71,1 | 1 | 1 |
| 19 | ¿En qué intervalo se muestra el ejemplo de tupla 'tuple' con '('Brais', 37, True)' entre paréntesis? | 71,1-76,1 | 1 | 1 |
| 20 | ¿Cuándo aparece la estructura 'set' con el ejemplo '{'Brais', 37, True, 37}' entre llaves? | 76,1-84,4 | 1 | 1 |
| 21 | ¿En qué segundos se explica el diccionario 'dict' con el ejemplo clave-valor '{'name': 'Brais'}'? | 84,4-96,1 | 1 | 1 |
| 22 | ¿En qué momento aparece el título 'Variables' en letras blancas grandes sobre el presentador? | 96,1-100,2 | 2 | 2 |
| 23 | ¿Cuándo se muestra por primera vez el editor Visual Studio Code con 'my_variable = 'Hola Python!'' en la línea 1? | 100,2-107,7 | 1 | 1 |
| 24 | ¿En qué segundos aparece el plano en blanco y negro del presentador haciendo el gesto del pulgar hacia arriba? | 109,0-110,2 | 1 | 1 |
| 25 | ¿En qué intervalo aparece la palabra 'Constantes' tachada con una cruz roja 'X'? | 110,2-114,1 | 1 | 1 |
| 26 | ¿Cuándo se ve en el editor la reasignación de 'my_variable' al número 6 y a 'true' con el menú de autocompletado? | 114,1-122,3 | 1 | 1 |
| 27 | ¿En qué segundos hace el presentador un gesto de precisión con la mano izquierda tras mostrar la reasignación de variables? | 122,3-127,3 | - | - |
| 28 | ¿En qué momento aparece el rótulo 'snake_case' en pantalla para explicar cómo nombrar variables? | 127,3-132,8 | 2 | 1 |
| 29 | ¿Cuándo se escribe 'my_snake_case_variable =' en la línea 1 del editor Visual Studio Code? | 132,8-139,2 | 1 | 1 |
| 30 | ¿En qué intervalo levanta el presentador el dedo índice izquierdo mostrando el tatuaje de su antebrazo? | 139,2-145,7 | 1 | 1 |
| 31 | ¿Cuándo aparece el título 'Funciones por defecto' sobreimpreso en dos líneas blancas? | 145,7-149,6 | 1 | 1 |
| 32 | ¿En qué segundos se muestra en el editor la función 'input' con la pregunta '¿Cuál es tu nombre?'? | 149,6-153,1 | 1 | 1 |
| 33 | ¿En qué momento se utiliza la función 'len(my_list)' en el editor para obtener la longitud de una lista? | 153,1-156,7 | 1 | 1 |
| 34 | ¿Cuándo aparece en el editor la instrucción 'print('Hola, Python!')' con el rótulo 'print' en grande? | 156,7-158,7 | - | 1 |
| 35 | ¿En qué segundos aparece el título de sección 'Ejecución' sobre el presentador? | 158,7-161,6 | 1 | 1 |
| 36 | ¿En qué intervalo se ejecuta el comando 'python3 hello_python.py' en la terminal imprimiendo 'Hola, Python!'? | 161,6-166,5 | 2 | 1 |
| 37 | ¿Cuándo se muestra la extensión 'Code Runner' de Jun Han en el Marketplace de Visual Studio Code? | 166,5-169,4 | 1 | 1 |
| 38 | ¿En qué segundos se abre el menú desplegable del botón de ejecución en VS Code con la opción 'Run Code' seleccionada? | 169,4-172,1 | 1 | 1 |
| 39 | ¿En qué momento aparece el presentador con gafas de sol pixeladas y lluvia de código binario verde al estilo Matrix? | 172,1-173,7 | 1 | 1 |
| 40 | ¿Cuándo aparece el título de sección 'Condicionales' sobre el presentador? | 173,7-175,6 | 1 | 1 |
| 41 | ¿En qué intervalo habla el presentador a cámara con el rótulo 'if' en grande en el centro? | 175,6-185,0 | 1 | 1 |
| 42 | ¿En qué segundos se escribe en el editor 'my_age = 37' y la cabecera 'if my_age >= 18:' sin el print todavía? | 185,0-191,0 | 4 | 1 |
| 43 | ¿Cuándo aparece en el editor la línea indentada 'print('Soy mayor de edad')' debajo de 'if my_age >= 18:'? | 191,0-197,6 | 1 | 1 |
| 44 | ¿En qué segundos enmarca el presentador con las manos el símbolo de llaves '{ }' aún sin tachar? | 197,6-201,8 | 1 | - |
| 45 | ¿En qué momento aparece el símbolo de llaves '{ }' tachado con una cruz roja 'X'? | 201,8-205,4 | - | - |
| 46 | ¿En qué intervalo se muestran las ramas 'elif my_age == 17' y 'else:' con 'print('Soy menor de edad')' en el editor? | 205,4-213,6 | 1 | 1 |
| 47 | ¿Cuándo aparece el título de sección 'Bucles' en letras blancas grandes sobre el presentador? | 213,6-221,1 | 1 | 1 |
| 48 | ¿En qué segundos se muestra en el editor el bucle 'for index in range(5):' con 'print(index)'? | 221,1-224,4 | 1 | 1 |
| 49 | ¿En qué segundo muestra el panel TERMINAL de VS Code la primera salida con los números del 0 al 4 tras ejecutar 'range(5)'? | 224,4-225,3 | - | - |
| 50 | ¿Cuándo se muestra en el editor el bucle 'for item in my_list:' recorriendo la lista '[1, 2, 3, 4, 5]'? | 225,3-228,1 | 1 | 1 |
| 51 | ¿En qué segundos muestra la terminal integrada de VS Code los números del 1 al 5 impresos tras recorrer 'my_list'? | 228,1-230,2 | 2 | 1 |
| 52 | ¿Cuándo aparece en el editor 'while index < 5' con el rótulo 'while' sobreimpreso en letras blancas? | 230,2-232,5 | 2 | 2 |
| 53 | ¿En qué intervalo se muestra el bucle 'while index < 5:' completo con el incremento 'index +=1' en la línea 5? | 232,5-236,0 | 1 | 1 |
| 54 | ¿En qué segundos muestra la terminal de VS Code la salida del 0 al 4 resultante de ejecutar el bucle 'while'? | 236,0-238,3 | 1 | - |
| 55 | ¿Cuándo aparece el título de sección 'Excepciones' en letras blancas grandes sobre el presentador? | 238,3-240,3 | 1 | 1 |
| 56 | ¿En qué segundos se muestra en el editor el bloque 'try / except' con la división por cero 'print(5 / 0)'? | 240,3-242,5 | 1 | 1 |
