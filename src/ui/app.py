"""Interfaz de usuario interactiva (Streamlit) para el TFM Video RAG Multimodal."""

from __future__ import annotations

import asyncio
import atexit
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    AUDIT_DIR,
    DB_PATH,
    RAW_DIR,
    VECTOR_SIZE,
    build_run_config,
)
from src.models import (
    QuotaSafetyError,
    RerankedResult,
    SearchResult,
    VectorDBError,
    VectorizationError,
)
from src.rag_engine.service import VideoRAGService, format_timestamp
from src.utils.audit import RunAuditor

logger = logging.getLogger(__name__)


def build_ui_run_config() -> dict[str, Any]:
    """Construye la configuración de la sesión UI para auditoría."""
    return build_run_config(
        mode="streamlit_ui",
        raw_dir=RAW_DIR,
        db_path=DB_PATH,
        vector_size=VECTOR_SIZE,
    )


@st.cache_resource
def get_auditor() -> RunAuditor:
    """Instancia única del auditor para la sesión de Streamlit."""
    auditor = RunAuditor(AUDIT_DIR)
    auditor.__enter__()
    atexit.register(lambda: auditor.__exit__(None, None, None))
    auditor.save_config(build_ui_run_config())
    logger.info("Auditoría de Streamlit UI iniciada")
    return auditor


@st.cache_resource
def get_rag_service() -> VideoRAGService:
    """Instancia persistente de la fachada VideoRAGService para la sesión UI."""
    return VideoRAGService(
        db_path=DB_PATH,
        raw_dir=RAW_DIR,
        vector_size=VECTOR_SIZE,
    )


async def execute_search(
    query: str,
    service: VideoRAGService,
    limit: int,
    video_filter: str | None = None,
    auditor: RunAuditor | None = None,
    use_reranker: bool = True,
) -> list[SearchResult] | list[RerankedResult]:
    """Delega la búsqueda en dos etapas y el registro de auditoría en ``VideoRAGService``."""
    results = await service.search(
        query,
        limit=limit,
        video_filter=video_filter,
        rerank=use_reranker,
        auditor=auditor,
    )
    return results[:limit]


def index_video_in_ui(
    video_path: Path,
    service: VideoRAGService,
    max_concurrent: int = 3,
    threshold: float = 0.5,
    min_shot_frames: int = 12,
    entropy_stride: int = 1,
    jpeg_quality: int = 80,
    output_dimensionality: int = VECTOR_SIZE,
    target_width: int = 512,
) -> int:
    """Ejecuta el pipeline de indexación mostrando avance en vivo en la interfaz."""
    progress_bar = st.progress(0.0, text="Iniciando pipeline de procesamiento...")

    def progress_callback(status: str, fraction: float) -> None:
        clamped = max(0.0, min(1.0, fraction))
        progress_bar.progress(clamped, text=status)

    try:
        inserted = asyncio.run(
            service.index_video(
                video_path,
                max_concurrent=max_concurrent,
                threshold=threshold,
                min_shot_frames=min_shot_frames,
                entropy_stride=entropy_stride,
                jpeg_quality=jpeg_quality,
                output_dimensionality=output_dimensionality,
                target_width=target_width,
                progress_callback=progress_callback,
            )
        )
        progress_bar.progress(1.0, text=f"Completado: {inserted} chunks indexados.")
        return inserted
    except (QuotaSafetyError, VectorizationError, VectorDBError, ValueError):
        progress_bar.empty()
        raise


def render_sidebar(
    service: VideoRAGService,
    auditor: RunAuditor,
) -> tuple[str | None, int, bool, bool]:
    """Renderiza la barra lateral con gestión de vídeos y parámetros."""
    st.sidebar.header("Gestión de Vídeos")

    indexed_videos = []
    try:
        indexed_videos = service.get_indexed_videos()
    except (VectorDBError, OSError) as exc:
        logger.warning("No se pudieron leer los vídeos indexados: %s", exc)

    selected_video: str | None = None
    if indexed_videos:
        default_idx = 0
        current_active = st.session_state.get("active_video")
        if current_active in indexed_videos:
            default_idx = indexed_videos.index(current_active)

        selected_video = st.sidebar.selectbox(
            "Seleccionar vídeo de trabajo:",
            options=indexed_videos,
            index=default_idx,
            help="Vídeos indexados en Qdrant disponibles para consulta.",
        )
        if selected_video:
            st.sidebar.caption("Estado: Indexado en Qdrant")
    else:
        st.sidebar.info("No hay vídeos indexados en la base de datos.")

    with st.sidebar.expander("Parámetros de Indexación", expanded=False):
        st.markdown("**Fragmentación SSIM**")
        p_threshold = st.slider(
            "Umbral SSIM",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help="Menor umbral = planos más largos y menos chunks (ahorro de cuota).",
        )
        p_min_shot_frames = st.slider(
            "Fotogramas mínimos por plano",
            min_value=5,
            max_value=90,
            value=12,
            step=5,
            help="12 fotogramas = 0.4s a 30fps. Subir a 30-45 evita microcortes.",
        )
        p_entropy_stride = st.slider(
            "Paso de muestreo de entropía",
            min_value=1,
            max_value=5,
            value=1,
            help="1 analiza todos los frames. 2 analiza 1 de cada 2 frames.",
        )

        st.markdown("**Vectorización y Red**")
        p_max_concurrent = st.slider(
            "Concurrencia máxima API",
            min_value=1,
            max_value=10,
            value=3,
            help="Peticiones simultáneas a Gemini (3 para capa gratuita de 30K TPM).",
        )
        p_jpeg_quality = st.slider(
            "Calidad JPEG fotogramas",
            min_value=50,
            max_value=95,
            value=80,
            step=5,
            help="Calidad de las imágenes enviadas. 80 ahorra un 35% de ancho de banda.",
        )
        p_target_width = st.slider(
            "Ancho máx. fotogramas (px)",
            min_value=256,
            max_value=1280,
            value=512,
            step=64,
            help="Redimensiona los fotogramas clave antes de enviarlos a Gemini. 512px reduce ~60% los tokens y previene el error 429.",
        )

    if selected_video and st.sidebar.button(
        "Re-indexar vídeo seleccionado", use_container_width=True
    ):
        target_path = RAW_DIR / selected_video
        if target_path.is_file():
            with st.spinner(f"Re-indexando {selected_video}..."):
                try:
                    n = index_video_in_ui(
                        target_path,
                        service,
                        max_concurrent=p_max_concurrent,
                        threshold=p_threshold,
                        min_shot_frames=p_min_shot_frames,
                        entropy_stride=p_entropy_stride,
                        jpeg_quality=p_jpeg_quality,
                        target_width=p_target_width,
                    )
                    st.sidebar.success(f"Re-indexados {n} chunks de '{selected_video}'.")
                    st.session_state["active_video"] = selected_video
                    st.rerun()
                except (
                    QuotaSafetyError,
                    VectorizationError,
                    VectorDBError,
                    ValueError,
                    OSError,
                ) as e:
                    st.sidebar.error(f"Error al re-indexar: {e}")
        else:
            st.sidebar.error(f"El archivo {selected_video} no existe en data/raw/.")

    raw_files = [f.name for f in RAW_DIR.glob("*.mp4")] if RAW_DIR.exists() else []
    unindexed_raw = sorted([f for f in raw_files if f not in indexed_videos])

    if unindexed_raw:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Indexar archivo local en disco")
        chosen_raw = st.sidebar.selectbox(
            "Vídeos sin indexar detectados en data/raw/:",
            options=unindexed_raw,
            help="Archivos presentes en la carpeta local data/raw/ que aún no tienen vectores en Qdrant.",
        )
        if st.sidebar.button(f"Indexar '{chosen_raw}'", use_container_width=True):
            target_path = RAW_DIR / chosen_raw
            with st.spinner(f"Indexando {chosen_raw}..."):
                try:
                    n = index_video_in_ui(
                        target_path,
                        service,
                        max_concurrent=p_max_concurrent,
                        threshold=p_threshold,
                        min_shot_frames=p_min_shot_frames,
                        entropy_stride=p_entropy_stride,
                        jpeg_quality=p_jpeg_quality,
                        target_width=p_target_width,
                    )
                    st.sidebar.success(f"Indexados {n} chunks de '{chosen_raw}'.")
                    st.session_state["active_video"] = chosen_raw
                    st.rerun()
                except (
                    QuotaSafetyError,
                    VectorizationError,
                    VectorDBError,
                    ValueError,
                    OSError,
                ) as e:
                    st.sidebar.error(f"Error al indexar: {e}")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Subir nuevo vídeo")
    uploaded_file = st.sidebar.file_uploader(
        "Carga un vídeo (.mp4)",
        type=["mp4", "avi", "mov", "mkv"],
        help="El archivo se guardará en data/raw/ para ser procesado.",
    )

    if uploaded_file is not None:
        dest_path = RAW_DIR / uploaded_file.name
        if st.sidebar.button(
            f"Guardar e Indexar '{uploaded_file.name}'", use_container_width=True
        ):
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Archivo guardado en data/raw/{uploaded_file.name}")

            with st.spinner(f"Indexando {uploaded_file.name}..."):
                try:
                    n = index_video_in_ui(
                        dest_path,
                        service,
                        max_concurrent=p_max_concurrent,
                        threshold=p_threshold,
                        min_shot_frames=p_min_shot_frames,
                        entropy_stride=p_entropy_stride,
                        jpeg_quality=p_jpeg_quality,
                        target_width=p_target_width,
                    )
                    st.sidebar.success(f"Completado: {n} chunks indexados.")
                    st.session_state["active_video"] = uploaded_file.name
                    st.rerun()
                except (
                    QuotaSafetyError,
                    VectorizationError,
                    VectorDBError,
                    ValueError,
                    OSError,
                ) as e:
                    st.sidebar.error(f"Error en el proceso: {e}")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Configuración de Búsqueda")
    top_k = st.sidebar.slider(
        "Top K resultados",
        min_value=1,
        max_value=10,
        value=5,
        help="Resultados finales mostrados. Con Etapa 2 activa, Qdrant recupera el doble de candidatos (2x) antes de reordenar.",
    )
    enable_reranker = st.sidebar.checkbox(
        "Reordenación Multimodal (Etapa 2)",
        value=True,
        help="Verifica los fotogramas de los candidatos con Gemini 3.5 Flash Lite para descartar falsos positivos.",
    )
    filter_by_selected = False
    if selected_video:
        filter_by_selected = st.sidebar.checkbox(
            f"Buscar solo en '{selected_video}'",
            value=False,
            help="Si no se marca, la búsqueda compara contra todos los vídeos indexados en Qdrant.",
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Mantenimiento de la Base de Datos")
    if (
        selected_video
        and selected_video in indexed_videos
        and st.sidebar.button(
            f"Eliminar '{selected_video}' de la BD", use_container_width=True
        )
    ):
        try:
            service.delete_video(selected_video)
            st.sidebar.success(f"Vectores de '{selected_video}' eliminados.")
            st.rerun()
        except (VectorDBError, OSError) as e:
            st.sidebar.error(f"Error al eliminar: {e}")

    if st.sidebar.button("Limpiar Base de Datos Completa", use_container_width=True):
        try:
            service.clear_database()
            st.session_state["last_results"] = []
            st.sidebar.success("Base de datos vectorial vaciada por completo.")
            st.rerun()
        except (VectorDBError, OSError) as e:
            st.sidebar.error(f"Error al limpiar la BD: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Auditoría de Sesión")
    st.sidebar.caption(
        f"Carpeta activa:\n`audit/{auditor.run_dir.name}`\n\n"
        "Registra logs de ejecución, configuración y búsquedas en tiempo real."
    )

    return selected_video, top_k, filter_by_selected, enable_reranker


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="TFM Video RAG Multimodal",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(120deg, #2563eb, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">Video RAG Multimodal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Recuperación semántica de eventos en vídeo mediante '
        "embeddings multimodales continuos (Google Gemini Embedding 2 + Qdrant)</div>",
        unsafe_allow_html=True,
    )

    service = get_rag_service()
    auditor = get_auditor()
    selected_video, top_k, filter_by_selected, enable_reranker = render_sidebar(
        service, auditor
    )

    if "play_time" not in st.session_state:
        st.session_state["play_time"] = 0
    if selected_video:
        st.session_state["active_video"] = selected_video
    else:
        st.session_state["active_video"] = None
        st.session_state["last_results"] = []
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("Reproductor Interactivo")
        active_video = st.session_state.get("active_video")

        if active_video:
            video_file_path = RAW_DIR / active_video
            if video_file_path.is_file():
                play_time = int(st.session_state.get("play_time", 0))
                st.caption(
                    f"**Vídeo en pantalla:** `{active_video}` | "
                    f"Posición actual: **{format_timestamp(play_time)}** ({play_time}s)"
                )
                st.video(str(video_file_path), start_time=play_time)
            else:
                st.warning(
                    f"El archivo `{active_video}` no se encuentra en el directorio `data/raw/`."
                )
        else:
            st.info(
                "No hay vídeos indexados en la base de datos. Indexa un archivo o sube uno nuevo desde la barra lateral."
            )

    with col_right:
        st.subheader("Consulta en Lenguaje Natural")

        query_input = st.text_input(
            "¿Qué momento o evento deseas localizar?",
            placeholder="Ej: un león rugiendo, bucles for, pingüinos en el agua...",
            help="Escribe en lenguaje natural. Se transformará en un embedding semántico denso.",
            disabled=active_video is None,
        )

        col_btn1, _ = st.columns([0.4, 0.6])
        search_clicked = col_btn1.button(
            "Buscar Evento",
            type="primary",
            use_container_width=True,
            disabled=active_video is None,
        )

        if search_clicked and query_input.strip():
            st.session_state["last_query"] = query_input.strip()
            video_filter = active_video if filter_by_selected else None

            spinner_msg = (
                "Etapa 1: Búsqueda vectorial + Etapa 2: Verificación visual con Gemini 3.5 Flash Lite..."
                if enable_reranker
                else "Generando embedding de texto y consultando Qdrant..."
            )
            with st.spinner(spinner_msg):
                try:
                    results = asyncio.run(
                        execute_search(
                            query=query_input.strip(),
                            service=service,
                            limit=top_k,
                            video_filter=video_filter,
                            auditor=auditor,
                            use_reranker=enable_reranker,
                        )
                    )
                    st.session_state["last_results"] = results
                except (VectorizationError, VectorDBError, ValueError, OSError) as exc:
                    st.error(f"Error al realizar la búsqueda: {exc}")

        last_results: list[Any] = st.session_state.get("last_results", [])
        last_query = st.session_state.get("last_query", "")

        if last_results:
            st.markdown(f"#### Resultados para: *\"{last_query}\"*")

            for rank, r in enumerate(last_results[:top_k], start=1):
                with st.container(border=True):
                    c1, c2 = st.columns([0.68, 0.32])

                    with c1:
                        st.markdown(
                            f"**#{rank} | {r.video_name}**  \n"
                            f"**Intervalo:** **{format_timestamp(r.start_time)} - {format_timestamp(r.end_time)}** "
                            f"`({r.start_time:.1f}s - {r.end_time:.1f}s)`"
                        )

                        if isinstance(r, RerankedResult):
                            if r.is_relevant:
                                st.markdown(
                                    f'<span style="background-color:#064e3b;color:#6ee7b7;'
                                    f'padding:3px 8px;border-radius:4px;font-size:0.82rem;font-weight:600;">'
                                    f"Validado por VLM ({int(r.relevance_score * 100)}%)</span>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<span style="background-color:#7f1d1d;color:#fca5a5;'
                                    f'padding:3px 8px;border-radius:4px;font-size:0.82rem;font-weight:600;">'
                                    f"Descartado / Falso Positivo ({int(r.relevance_score * 100)}%)</span>",
                                    unsafe_allow_html=True,
                                )

                            st.caption(
                                f"**Análisis visual (Gemini 3.5 Flash Lite):** {r.reasoning}"
                            )

                            score_pct = max(0, min(100, int(r.relevance_score * 100)))
                            st.progress(
                                score_pct / 100.0,
                                text=f"Relevancia visual: {r.relevance_score:.2f} | Coseno Qdrant: {r.vector_score:.4f}",
                            )
                        else:
                            score_pct = max(0, min(100, int(r.score * 100)))
                            st.progress(
                                score_pct / 100.0, text=f"Similitud coseno: {r.score:.4f}"
                            )

                    with c2:
                        btn_key = f"play_btn_{r.video_name}_{r.chunk_index}_{rank}"
                        if st.button("Saltar aquí", key=btn_key, use_container_width=True):
                            st.session_state["active_video"] = r.video_name
                            st.session_state["play_time"] = int(r.start_time)
                            st.rerun()

        elif last_query and not search_clicked:
            st.info("No se encontraron fragmentos similares para esta consulta.")


if __name__ == "__main__":
    main()
