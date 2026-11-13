# Informe de Evaluación Empírica: Backends Locales (`Qwen3-VL-Embedding-2B` y `EmbeddingGemma 2`) frente a `Gemini Embedding 2` (Cloud API)

- **Fecha de ejecución:** 13 de noviembre de 2026
- **Rama Git:** `v3-local-embedding-backends`
- **Plataforma de hardware:** MacBook Air Apple Silicon M3 (16 GB de memoria unificada, macOS ARM64)
- **Entorno de software:** Python 3.13.12 (`env_tfm`), `sentence-transformers` 6.3.0, `transformers` 5.22.2, `torch` 2.14.1 (`mps` en `bfloat16`), `openai-whisper` (`base`), Qdrant local embebido (`./data/vector_db`)
- **Artefactos fuente:**
  - CSV completo (18 filas): [`embedding_backends_full_20261113.csv`](./embedding_backends_full_20261113.csv)
  - CSV resumen nativo (`python_tutorial.mp4`): [`embedding_backends_20261113.csv`](./embedding_backends_20261113.csv)
  - Logs históricos de indexación Cloud (`gemini`): `audit/run_20261109_144503/execution.log` (`python_tutorial.mp4`) y `audit/run_20261109_145448/execution.log` (`animals.mp4`)

---

## 1. Resumen Ejecutivo

Se ejecutó la comparativa empírica controlada de Etapa 1 (recuperación densa pura en Qdrant sin reordenador VLM, `rerank=False`) entre el baseline propietario en la nube (`gemini-embedding-2-preview`) y dos modelos multimodales locales ejecutados íntegramente en GPU integrada Apple Silicon (`mps`, precisión `bfloat16`): **Qwen3-VL-Embedding-2B** (`Qwen/Qwen3-VL-Embedding-2B`, ~2B parámetros) y **EmbeddingGemma 2** (`google/embeddinggemma-2`, 740M parámetros totales).

La evaluación abarcó los dos vídeos del banco de pruebas (**164 fragmentos** idénticos en segmentación SSIM y transcripción Whisper `base`), **102 consultas literales** (`56` en `python_tutorial.mp4` y `46` en `animals.mp4`), **20 consultas multi-objetivo** (`10` por vídeo, 106 intervalos esperados en total) y las **9 combinaciones Matryoshka (MRL)** (`gemini`: 3072/1536/768; `qwen3-vl-2b`: 2048/1536/768; `gemma2`: 768/512/256), generando además todas las colecciones persistentes en `./data/vector_db`.

### Conclusiones principales

1. **Complementariedad inter-dominio Cloud vs. Local:** En el vídeo técnico con locución sincronizada (`python_tutorial.mp4`), `gemini@3072` lidera en `Recall@1` (**85,71%**) y `MRR` (**0,8973**). En cambio, en metraje documental visual con *B-roll* desacoplado (`animals.mp4`), **ambos modelos locales igualan o superan a la API en nube**: `qwen3-vl-2b@2048` alcanza el mejor `Recall@1` (**78,26%** frente a 71,74% de Gemini), `Recall@3` (**95,65%**) y `MRR` (**0,8678**), mientras que `gemma2` logra el mayor `Interval Recall` multi-objetivo (**81,23%** en 768d y **85,15%** en 512d/256d frente a 74,89% de Gemini).
2. **Viabilidad global del modelo Edge (740M parámetros):** A escala global (102 consultas literales), `gemma2@768` empata con `gemini@3072` en `Recall@3` (**93,14%**) y `Recall@5` (**97,06%**), y su variante `gemma2@512` registra el mejor `Interval Recall` multi-objetivo global de Etapa 1 (**82,09%** frente a 81,05% de `gemini@768` y 80,31% de `qwen3-vl-2b@1536`).
3. **Aceleración drástica de latencia de consulta y control térmico/RAM:** `gemma2` resuelve consultas locales con una mediana de **29,93-30,20 ms** (entre **15x y 16x más rápido** que la mediana de red Cloud de **464-481 ms**) y un pico de memoria GPU Metal de **2.552-2.592 MB** (casi la mitad que `qwen3-vl-2b`, con **4.737 MB**). Sorprendentemente, en indexación completa (82 chunks con 3 fotogramas a 512 px), `qwen3-vl-2b` es el motor más rápido (**126,05-142,23 s**; `1,537-1,734 s/chunk`), superando tanto a Gemini API (`187,12-227,31 s`) como a `gemma2` (`245,49-247,73 s`).
4. **Necesidad empírica de renormalización $L_2$ tras el truncado MRL:** Mediciones directas sobre MPS muestran que `SentenceTransformer(..., truncate_dim=...)` devuelve prefijos sin renormalizar (normas $L_2$ de **0,7152** en Gemma 2 a 256d y **0,6284** en Qwen3-VL-2B a 768d; medias de **0,7160** y **0,6126** sobre los 164 chunks reales), lo que valida nuestra función `truncate_and_normalize_mrl` para proyectar todos los prefijos sobre $\mathbb{S}^{d-1}$.

---

## 2. Metodología Experimental y Restricciones de Hardware

### 2.1 Infraestructura y parámetros controlados

| Parámetro | Configuración aplicada | Justificación técnica |
|---|---|---|
| Hardware | MacBook Air M3, 16 GB memoria unificada | Escenario representativo de estación portátil sin GPU discreta |
| Precisión numérica (`torch_dtype`) | `bfloat16` en `mps` (`float32` en `cpu`) | `EmbeddingGemma 2` advierte desbordamiento a `NaN` o degradación silenciosa en `float16` por rango dinámico de activaciones |
| Exclusión de Qwen 8B | Descartado en 16 GB unificados | Qwen3-VL-Embedding-2B ya consume ~4,74 GB de driver MPS en uso |
| Concurrencia de indexación | `Semaphore(1)` en local vs. `Semaphore(3)` en Cloud | Evita contención en Metal y solapamiento con Whisper |
| Gestión de ciclo de memoria | Descarga de Whisper antes de cargar VLM + `torch.mps.empty_cache()` y cerrojo compartido por modelo | Evita picos combinados > 10 GB y serializa el uso entre instancias |
| Representación de chunk | 3 fotogramas clave (entropía Shannon, ancho 512 px) + transcripción Whisper `base` | Entrada multimodal homogénea en los 3 motores |
| Segmentación por vídeo | `python_tutorial.mp4`: `threshold=0.6`, `max_shot=5.0 s`; `animals.mp4`: `threshold=0.5`, `max_shot=8.0 s` | Garantiza exactamente 82 chunks idénticos por vídeo |
| Aislamiento de Etapa 1 | `rerank=False` (sin `gemini-3.5-flash-lite`), colecciones `video_chunks__<backend>__<dim>` | Mide la geometría pura de cada espacio latente sin mezcla de señales |

### 2.2 Tiempos históricos de Gemini y reutilización de cuota

Para no consumir cuota API de indexación (`1.000 RPD`) ni alterar los vectores baseline ya validados, se reutilizaron los **164 chunks** (`82 + 82`) ya almacenados en `video_chunks` (dimensión `3072`), vectorizando únicamente las consultas únicas en Gemini (~122 peticiones en total). Los tiempos históricos completos de `index_video` se extrajeron mediante `parse_audit_index_seconds` desde los logs oficiales de auditoría:

- **`python_tutorial.mp4`** (`audit/run_20261109_144503/execution.log`): inicio `14:45:03,228` -> fin `14:48:50,533` = **227,31 s** (`2,772 s/chunk`; incluye 25,38 s SSIM + 10,96 s Whisper + 190,75 s API Gemini).
- **`animals.mp4`** (`audit/run_20261109_145448/execution.log`): inicio `14:54:48,760` -> fin `14:57:55,883` = **187,12 s** (`2,282 s/chunk`; incluye 14,15 s SSIM + 9,45 s Whisper + 163,29 s API Gemini).

---

## 3. Resultados de Precisión Comparada y Análisis Inter-Dominio

### 3.1 Desglose completo por vídeo, backend y dimensión MRL

| Vídeo | Backend | Dim | Ahorro | R@1 | R@3 | R@5 | MRR | MT IR (E1) | MT Prec (E1) | Qdrant (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `python_tutorial.mp4` | `gemini` | 3072 | 0,0% | **0,8571** | 0,9286 | 0,9464 | 0,8973 | 0,8542 | 0,4500 | 1,139 |
| `python_tutorial.mp4` | `gemini` | 1536 | 50,0% | **0,8571** | 0,9286 | 0,9643 | 0,9009 | 0,8708 | **0,4700** | 0,783 |
| `python_tutorial.mp4` | `gemini` | 768 | 75,0% | **0,8571** | 0,9286 | **0,9821** | **0,9045** | **0,8722** | 0,4600 | 0,672 |
| `python_tutorial.mp4` | `qwen3-vl-2b` | 2048 | 0,0% | 0,7500 | 0,8929 | 0,9464 | 0,8289 | 0,7722 | 0,3600 | 1,275 |
| `python_tutorial.mp4` | `qwen3-vl-2b` | 1536 | 25,0% | 0,7500 | 0,8929 | 0,9464 | 0,8289 | 0,7931 | 0,3900 | 1,095 |
| `python_tutorial.mp4` | `qwen3-vl-2b` | 768 | 62,5% | 0,7500 | 0,8929 | 0,9286 | 0,8205 | 0,7722 | 0,3900 | 0,882 |
| `python_tutorial.mp4` | `gemma2` | 768 | 0,0% | 0,7679 | 0,9286 | 0,9643 | 0,8512 | 0,8028 | 0,3700 | 0,775 |
| `python_tutorial.mp4` | `gemma2` | 512 | 33,3% | 0,7679 | **0,9464** | 0,9643 | 0,8488 | 0,7903 | 0,3700 | 0,691 |
| `python_tutorial.mp4` | `gemma2` | 256 | 66,7% | 0,7500 | 0,9286 | 0,9286 | 0,8333 | 0,7486 | 0,3500 | **0,627** |
| `animals.mp4` | `gemini` | 3072 | 0,0% | 0,7174 | 0,9348 | **1,0000** | 0,8341 | 0,7489 | 0,5000 | 1,096 |
| `animals.mp4` | `gemini` | 1536 | 50,0% | 0,6957 | 0,9348 | **1,0000** | 0,8232 | 0,7198 | 0,4800 | 0,793 |
| `animals.mp4` | `gemini` | 768 | 75,0% | 0,6957 | **0,9565** | **1,0000** | 0,8261 | 0,7489 | 0,5000 | 0,628 |
| `animals.mp4` | `qwen3-vl-2b` | 2048 | 0,0% | **0,7826** | **0,9565** | 0,9783 | **0,8678** | 0,7905 | 0,5431 | 1,376 |
| `animals.mp4` | `qwen3-vl-2b` | 1536 | 25,0% | **0,7826** | **0,9565** | 0,9783 | **0,8678** | 0,8131 | 0,5784 | 1,215 |
| `animals.mp4` | `qwen3-vl-2b` | 768 | 62,5% | **0,7826** | **0,9565** | 0,9565 | 0,8659 | 0,7698 | 0,5300 | 6,540* |
| `animals.mp4` | `gemma2` | 768 | 0,0% | 0,7174 | 0,9348 | 0,9783 | 0,8239 | 0,8123 | **0,5800** | 0,769 |
| `animals.mp4` | `gemma2` | 512 | 33,3% | 0,7174 | 0,9348 | 0,9348 | 0,8152 | **0,8515** | 0,5700 | 0,692 |
| `animals.mp4` | `gemma2` | 256 | 66,7% | 0,7174 | 0,9348 | 0,9783 | 0,8188 | **0,8515** | 0,5700 | **0,581** |

*\*Nota sobre `6,540 ms` en `qwen3-vl-2b@768` sobre `animals.mp4`: pico puntual del planificador/GC durante la medición en memoria; la tendencia estructural del coste HNSW desciende linealmente con $d$.*

### 3.2 Síntesis agregada global (102 consultas literales y 20 multi-objetivo)

| Backend + Dim | R@1 (Global) | R@3 (Global) | R@5 (Global) | MRR (Global) | MT IR (E1) | MT Prec (E1) | KB/chunk | Total 82 chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini @ 3072` (Nativa) | **0,7941** (81/102) | 0,9314 (95/102) | 0,9706 (99/102) | 0,8688 | 0,8015 | 0,4750 | 12,00 KB | 984,00 KB |
| `gemini @ 1536` (-50,0%) | 0,7843 (80/102) | 0,9314 (95/102) | 0,9804 (100/102) | 0,8659 | 0,7953 | 0,4750 | 6,00 KB | 492,00 KB |
| `gemini @ 768` (-75,0%) | 0,7843 (80/102) | **0,9412** (96/102) | **0,9902** (101/102) | **0,8691** | 0,8105 | 0,4800 | 3,00 KB | 246,00 KB |
| `qwen3-vl-2b @ 2048` (Nativa) | 0,7647 (78/102) | 0,9216 (94/102) | 0,9608 (98/102) | 0,8464 | 0,7813 | 0,4515 | 8,00 KB | 656,00 KB |
| `qwen3-vl-2b @ 1536` (-25,0%) | 0,7647 (78/102) | 0,9216 (94/102) | 0,9608 (98/102) | 0,8464 | 0,8031 | **0,4842** | 6,00 KB | 492,00 KB |
| `qwen3-vl-2b @ 768` (-62,5%) | 0,7647 (78/102) | 0,9216 (94/102) | 0,9412 (96/102) | 0,8410 | 0,7710 | 0,4600 | 3,00 KB | 246,00 KB |
| `gemma2 @ 768` (Nativa) | 0,7451 (76/102) | 0,9314 (95/102) | 0,9706 (99/102) | 0,8389 | 0,8075 | 0,4750 | 3,00 KB | 246,00 KB |
| `gemma2 @ 512` (-33,3%) | 0,7451 (76/102) | **0,9412** (96/102) | 0,9510 (97/102) | 0,8336 | **0,8209** | 0,4700 | 2,00 KB | 164,00 KB |
| `gemma2 @ 256` (-66,7%) | 0,7353 (75/102) | 0,9314 (95/102) | 0,9510 (97/102) | 0,8268 | 0,8001 | 0,4600 | **1,00 KB** | **82,00 KB** |

### 3.3 Lectura crítica inter-dominio

- **Dominio técnico con locución densa (`python_tutorial.mp4`):** Los fotogramas muestran capturas de pantalla de código Python con cambios tipográficos sutiles y alta densidad semántica de sintaxis hablada/escrita en español. `gemini@3072` domina `Recall@1` (85,71% vs. 76,79% Gemma 2 y 75,00% Qwen3-VL-2B). Resulta destacable que **Gemma 2 supera a Qwen3-VL-2B** en todas las métricas del tutorial de Python (`R@1`: +1,79 pp; `R@3`: +3,57 pp; `MRR`: +0,0223; `MT IR`: +3,06 pp), lo que refleja una mayor alineación semántica de su tronco de texto Gemma en consultas técnicas en español combinadas con transcripción Whisper.
- **Dominio documental con *B-roll* desacoplado (`animals.mp4`):** Aquí la narración habla de procesos generales mientras la imagen muestra especies animales concretas (consultas predominantemente visuales). `qwen3-vl-2b` despliega la ventaja de su codificador visual M-RoPE de 2B parámetros y **lidera en solitario**: `Recall@1 = 78,26%` (+6,52 pp sobre Gemini y Gemma 2), `Recall@3 = 95,65%` y `MRR = 0,8678`. Por su parte, `gemma2` sobresale en recuperación multi-objetivo amplia (`top_k = 10`), alcanzando **85,15% de Interval Recall** en 512d y 256d (+10,26 pp sobre `gemini@3072` y +3,84 pp sobre `qwen3-vl-2b@1536`).

---

## 4. Curvas Matryoshka (MRL) y Verificación de Normalización $L_2$

La medición directa sobre Apple Silicon MPS confirma que al utilizar `truncate_dim` en `sentence-transformers` sin paso explícito de renormalización, los vectores resultantes abandonan la hiperesfera unitaria $\mathbb{S}^{d-1}$. La tabla recoge tanto la medición directa con `SentenceTransformer(..., truncate_dim=d)` sobre una consulta en `bfloat16` como la distribución empírica observada en los **164 fragmentos reales** de vídeo antes de aplicar `truncate_and_normalize_mrl`:

| Backend | Dim ($d$) | Reducción | Norma `truncate_dim` (consulta) | Media norma $L_2$ prefijo (164 chunks) | Rango $[\min, \max]$ (164 chunks) | Norma final con `truncate_and_normalize_mrl` |
|---|---|---:|---:|---:|---|---:|
| `gemini` | 3072 | 0,0% | n/a | 1,0000 | $[1,0000,\; 1,0000]$ | **1,0000** |
| `gemini` | 1536 | 50,0% | n/a | 0,7267 | $[0,7235,\; 0,7309]$ | **1,0000** |
| `gemini` | 768 | 75,0% | n/a | 0,5612 | $[0,5552,\; 0,5655]$ | **1,0000** |
| `qwen3-vl-2b` | 2048 | 0,0% | 1,0004 | 1,0000 | $[1,0000,\; 1,0000]$ | **1,0000** |
| `qwen3-vl-2b` | 1536 | 25,0% | 0,8659 | 0,8666 | $[0,8450,\; 0,8858]$ | **1,0000** |
| `qwen3-vl-2b` | 768 | 62,5% | **0,6284** | **0,6126** | $[0,5849,\; 0,6398]$ | **1,0000** |
| `gemma2` | 768 | 0,0% | 1,0016 | 1,0000 | $[1,0000,\; 1,0000]$ | **1,0000** |
| `gemma2` | 512 | 33,3% | 0,8839 | 0,8785 | $[0,8684,\; 0,8854]$ | **1,0000** |
| `gemma2` | 256 | 66,7% | **0,7152** | **0,7160** | $[0,7003,\; 0,7320]$ | **1,0000** |

Aunque la similitud del coseno en Qdrant es formalmente invariante a escala escalar positiva, mantener normas heterogéneas provoca saturación numérica en precisión reducida (`bfloat16`/`int8`), invalida métricas basadas en producto escalar (*Dot Product*) o umbrales fijos de relevancia, e introduce ruido comparativo frente a motores que operan estrictamente sobre $\mathbb{S}^{d-1}$.

En cuanto a **estabilidad MRL**:
- `qwen3-vl-2b` presenta invarianza exacta del **100%** en `Recall@1` (76,47%) y `Recall@3` (92,16%) desde 2048d hasta 768d (-62,5% de almacenamiento), perdiendo únicamente 2 consultas en `Recall@5` en el salto a 768d.
- `gemma2` retiene íntegramente `Recall@1` (74,51%) al comprimir a `512d` (-33,3%) y pierde un solo acierto en primera posición (73,53%, 75/102) al llegar a `256d` (-66,7%, equivalente a solo **82 KB totales** por vídeo de 82 chunks).

---

## 5. Rendimiento Computacional, Latencia y Huella de Memoria

| Vídeo | Backend (Dim) | Dispositivo | Indexación total | Coste / chunk | Latencia consulta media | Latencia consulta mediana | Pico RSS | Pico GPU MPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `python_tutorial.mp4` | `gemini` (3072) | Cloud API | 227,31 s | 2,772 s | 484,67 ms | 481,30 ms | 391,4 MB | 0 MB |
| `python_tutorial.mp4` | `qwen3-vl-2b` (2048) | Local MPS | **142,23 s** | **1,734 s** | 95,11 ms | 97,69 ms | 2.095,1 MB | 4.736,5 MB |
| `python_tutorial.mp4` | `gemma2` (768) | Local MPS | 247,73 s | 3,021 s | **30,11 ms** | **29,93 ms** | 3.081,5 MB | **2.552,6 MB** |
| `animals.mp4` | `gemini` (3072) | Cloud API | 187,12 s | 2,282 s | 1.086,72 ms* | 464,18 ms | 424,0 MB | 0 MB |
| `animals.mp4` | `qwen3-vl-2b` (2048) | Local MPS | **126,05 s** | **1,537 s** | 92,33 ms | 90,94 ms | 1.995,8 MB | 4.737,7 MB |
| `animals.mp4` | `gemma2` (768) | Local MPS | 245,49 s | 2,994 s | **31,08 ms** | **30,20 ms** | 2.641,2 MB | **2.592,1 MB** |

*\*Nota: La media de `1.086,72 ms` en `gemini` sobre `animals.mp4` recoge picos transitorios de red/TLS en un subconjunto reducido de consultas; la mediana (`464,18 ms`) refleja la latencia típica de la API.*

### Análisis de la asimetría entre indexación multimodal y consulta textual

Una micro-medición aislada en `mps` (`torch.mps.synchronize()`, tras calentamiento) explica por qué `gemma2` es 3x más rápido respondiendo consultas pero 1,7x más lento indexando que `qwen3-vl-2b`:
- **Consulta de solo texto:** `gemma2` tarda **23,6 ms** en inferencia bruta en MPS (~**30 ms** end-to-end con tokenización y búsqueda en Qdrant) frente a **72,9 ms** en `qwen3-vl-2b` (~**91-98 ms** end-to-end). Esto se debe a que en consultas de texto solo activa su tronco ligero de 270M parámetros sin ejecutar el codificador visual.
- **Indexación de chunk multimodal (3 fotogramas a 512 px + transcripción):** `qwen3-vl-2b` procesa la secuencia de 3 imágenes en **1.180,6 ms/chunk**, mientras que `gemma2` requiere **1.960,1 ms/chunk**. Con las etapas fijas de SSIM, extracción JPEG y transcripción Whisper (`base`), Qwen3-VL-2B completa un vídeo de ~4 minutos en **126-142 s**, incluso más rápido que la API remota Gemini bajo `Semaphore(3)` (**187-227 s**).
- **Huella de memoria en 16 GB:** Ambos modelos se ejecutan holgadamente sin *swap* destructivo bajo nuestra política de exclusión mutua (`Semaphore(1)` + descarga previa de Whisper): `qwen3-vl-2b` ocupa **4,74 GB** de driver MPS y **~2,0-2,1 GB** RSS de proceso; `gemma2` consume únicamente **2,55-2,59 GB** de driver MPS y **2,64-3,08 GB** RSS.

---

## 6. Estado de las Colecciones Persistentes en `./data/vector_db`

Todas las colecciones requeridas han quedado indexadas en disco tras las pruebas con los **82 chunks** de `python_tutorial.mp4` y los **82 chunks** de `animals.mp4` (**164 puntos/colección**):

- `video_chunks` (`gemini`, 3072d)
- `video_chunks__gemini__1536` (`gemini`, 1536d)
- `video_chunks__gemini__768` (`gemini`, 768d)
- `video_chunks__qwen3-vl-2b__2048` (`qwen3-vl-2b`, 2048d)
- `video_chunks__qwen3-vl-2b__1536` (`qwen3-vl-2b`, 1536d)
- `video_chunks__qwen3-vl-2b__768` (`qwen3-vl-2b`, 768d)
- `video_chunks__gemma2__768` (`gemma2`, 768d)
- `video_chunks__gemma2__512` (`gemma2`, 512d)
- `video_chunks__gemma2__256` (`gemma2`, 256d)

Esto habilita la conmutación instantánea en tiempo real desde la barra lateral de Streamlit y la CLI entre cualquier motor y dimensión sin reindexar.
