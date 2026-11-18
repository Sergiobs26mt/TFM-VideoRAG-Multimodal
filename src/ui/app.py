"""Interfaz de usuario interactiva (Streamlit) para el TFM Video RAG Multimodal."""

from __future__ import annotations

import asyncio
import atexit
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    AUDIT_DIR,
    DB_PATH,
    DEFAULT_EMBEDDING_BACKEND,
    EMBEDDING_BACKENDS,
    RAW_DIR,
    VECTOR_SIZE,
    WHISPER_MODEL,
    build_run_config,
    default_dimension,
)
from src.embeddings.factory import device_label_for
from src.models import (
    IndexStats,
    MultimodalQuery,
    QuotaSafetyError,
    RerankedResult,
    SearchResult,
    VectorDBError,
    VectorizationError,
)
from src.rag_engine.service import VideoRAGService
from src.ui.components import (
    IndexingParams,
    render_attachment_preview,
    render_results_section,
    render_station_header,
    render_video_monitor,
)
from src.utils.audit import RunAuditor

logger = logging.getLogger(__name__)

BACKEND_UI_LABELS: dict[str, str] = {
    "gemini": "Cloud: Gemini",
    "qwen3-vl-2b": "Local: Qwen3-VL 2B",
    "gemma2": "Local: EmbeddingGemma 2",
}


def mrl_options(backend: str) -> tuple[int, ...]:
    """Devuelve las dimensiones MRL válidas para el backend en orden descendente."""
    return tuple(sorted(EMBEDDING_BACKENDS[backend].dimensions, reverse=True))


def build_ui_run_config(
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    vector_size: int = VECTOR_SIZE,
    index_stats: IndexStats | None = None,
) -> dict[str, Any]:
    """Construye la configuración de la sesión UI para auditoría."""
    return build_run_config(
        mode="streamlit_ui",
        raw_dir=RAW_DIR,
        db_path=DB_PATH,
        vector_size=vector_size,
        embedding_backend=backend,
        index_stats=index_stats,
    )


def save_embedding_config(auditor: RunAuditor, service: VideoRAGService) -> None:
    """Actualiza run_config.json con el motor activo y las últimas estadísticas."""
    auditor.save_config(
        build_ui_run_config(
            backend=service.embedding_backend,
            vector_size=service.vector_size,
            index_stats=service.last_index_stats,
        )
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


def _extract_uploaded_bytes(item: Any) -> bytes:
    """Extrae los bytes en memoria de un widget de subida o grabación de Streamlit."""
    if isinstance(item, bytes):
        return item
    if hasattr(item, "getvalue"):
        return bytes(item.getvalue())
    if hasattr(item, "read"):
        return bytes(item.read())
    return b""


def build_ui_multimodal_query(
    text_input: str = "",
    uploaded_txts: Sequence[Any] | None = None,
    uploaded_images: Sequence[Any] | None = None,
    recorded_audio: Any | None = None,
    uploaded_audios: Sequence[Any] | None = None,
) -> MultimodalQuery:
    """Construye un ``MultimodalQuery`` inmutable a partir de los controles de la UI."""
    cleaned_text = text_input.strip() if text_input and text_input.strip() else None

    txt_list: list[str] = []
    if uploaded_txts:
        for txt_item in uploaded_txts:
            raw = _extract_uploaded_bytes(txt_item)
            decoded = raw.decode("utf-8", errors="replace").strip()
            if decoded:
                txt_list.append(decoded)

    img_list: list[bytes] = []
    if uploaded_images:
        for img_item in uploaded_images:
            raw = _extract_uploaded_bytes(img_item)
            if raw:
                img_list.append(raw)

    audio_list: list[bytes] = []
    if recorded_audio is not None:
        mic_bytes = _extract_uploaded_bytes(recorded_audio)
        if mic_bytes:
            audio_list.append(mic_bytes)
    if uploaded_audios:
        for aud_item in uploaded_audios:
            raw = _extract_uploaded_bytes(aud_item)
            if raw:
                audio_list.append(raw)

    return MultimodalQuery(
        text=cleaned_text,
        txt_contents=tuple(txt_list),
        images=tuple(img_list),
        audio_bytes=tuple(audio_list),
    )


async def execute_search(
    query: str | MultimodalQuery,
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
    enable_audio: bool = True,
    whisper_model: str = WHISPER_MODEL,
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
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
                enable_audio=enable_audio,
                whisper_model=whisper_model,
                embedding_backend=embedding_backend,
                progress_callback=progress_callback,
            )
        )
        progress_bar.progress(1.0, text=f"Completado: {inserted} chunks indexados.")
        return inserted
    except (QuotaSafetyError, VectorizationError, VectorDBError, ValueError):
        progress_bar.empty()
        raise


def _safe_indexed_videos(service: VideoRAGService) -> list[str]:
    """Obtiene los vídeos indexados sin interrumpir la UI ante errores de Qdrant."""
    try:
        return service.get_indexed_videos()
    except (VectorDBError, OSError) as exc:
        logger.warning("No se pudieron leer los vídeos indexados: %s", exc)
        return []


def _run_indexing_flow(
    *,
    video_path: Path,
    video_name: str,
    status_label: str,
    success_message: str,
    error_prefix: str,
    service: VideoRAGService,
    auditor: RunAuditor,
    params: IndexingParams,
) -> None:
    """Ejecuta la indexación con indicador de estado, actualiza auditoría y refresca."""
    with st.status(status_label, expanded=True) as status:
        try:
            inserted = index_video_in_ui(
                video_path,
                service,
                max_concurrent=params.max_concurrent,
                threshold=params.threshold,
                min_shot_frames=params.min_shot_frames,
                entropy_stride=params.entropy_stride,
                jpeg_quality=params.jpeg_quality,
                output_dimensionality=params.output_dimensionality,
                target_width=params.target_width,
                enable_audio=params.enable_audio,
                whisper_model=params.whisper_model,
                embedding_backend=params.embedding_backend,
            )
            save_embedding_config(auditor, service)
            done_text = success_message.format(n=inserted)
            status.update(label=done_text, state="complete", expanded=False)
            st.session_state["active_video"] = video_name
            st.session_state["play_time"] = 0
            st.session_state["pending_toast"] = done_text
            st.rerun()
        except (
            QuotaSafetyError,
            VectorizationError,
            VectorDBError,
            ValueError,
            OSError,
        ) as exc:
            err_text = f"{error_prefix}: {exc}"
            status.update(label=err_text, state="error", expanded=True)
            st.error(err_text, icon=":material/error:")


@st.dialog("Ingesta e indexación de vídeo", width="large", icon=":material/upload_file:")
def open_ingest_dialog(
    service: VideoRAGService,
    auditor: RunAuditor,
    params: IndexingParams,
    indexed_videos: Sequence[str],
    selected_video: str | None,
) -> None:
    """Diálogo modal para indexar archivos locales de data/raw/ o subir uno nuevo."""
    st.caption(
        f"Destino: {BACKEND_UI_LABELS[params.embedding_backend]} · "
        f"MRL {params.output_dimensionality}d"
    )

    tab_local, tab_upload = st.tabs(
        [
            ":material/folder_open: Archivo en data/raw/",
            ":material/cloud_upload: Subir nuevo vídeo",
        ]
    )

    with tab_local:
        raw_files = [f.name for f in RAW_DIR.glob("*.mp4")] if RAW_DIR.exists() else []
        unindexed_raw = sorted([f for f in raw_files if f not in indexed_videos])

        if unindexed_raw:
            chosen_raw = st.selectbox(
                "Vídeos pendientes en data/raw/",
                options=unindexed_raw,
                help="Archivos presentes en data/raw/ sin vectores en la colección activa.",
            )
            if st.button(
                f"Indexar '{chosen_raw}'",
                icon=":material/upload_file:",
                type="primary",
                key="dialog_index_local_btn",
                width="stretch",
            ):
                _run_indexing_flow(
                    video_path=RAW_DIR / chosen_raw,
                    video_name=chosen_raw,
                    status_label=f"Indexando {chosen_raw}...",
                    success_message=f"Indexados {{n}} chunks de '{chosen_raw}'.",
                    error_prefix="Error al indexar",
                    service=service,
                    auditor=auditor,
                    params=params,
                )
        else:
            st.caption("No hay archivos .mp4 pendientes de indexación en data/raw/.")

        if selected_video:
            target_path = RAW_DIR / selected_video
            if st.button(
                f"Re-indexar vídeo activo ('{selected_video}')",
                icon=":material/restart_alt:",
                key="dialog_reindex_active_btn",
                width="stretch",
            ):
                if target_path.is_file():
                    _run_indexing_flow(
                        video_path=target_path,
                        video_name=selected_video,
                        status_label=f"Re-indexando {selected_video}...",
                        success_message=f"Re-indexados {{n}} chunks de '{selected_video}'.",
                        error_prefix="Error al re-indexar",
                        service=service,
                        auditor=auditor,
                        params=params,
                    )
                else:
                    st.error(
                        f"El archivo {selected_video} no existe en data/raw/.",
                        icon=":material/error:",
                    )

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Selecciona un archivo de vídeo",
            type=["mp4", "avi", "mov", "mkv"],
            help="El archivo se guardará en data/raw/ para ser procesado.",
            key="dialog_video_uploader",
        )
        if uploaded_file is not None:
            dest_path = RAW_DIR / uploaded_file.name
            if st.button(
                f"Guardar e indexar '{uploaded_file.name}'",
                icon=":material/cloud_upload:",
                type="primary",
                key="dialog_upload_index_btn",
                width="stretch",
            ):
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                _run_indexing_flow(
                    video_path=dest_path,
                    video_name=uploaded_file.name,
                    status_label=f"Indexando {uploaded_file.name}...",
                    success_message="Completado: {n} chunks indexados.",
                    error_prefix="Error en el proceso",
                    service=service,
                    auditor=auditor,
                    params=params,
                )


@st.dialog("Mantenimiento de base de datos", icon=":material/warning:")
def open_maintenance_dialog(
    service: VideoRAGService,
    indexed_videos: Sequence[str],
    selected_video: str | None,
) -> None:
    """Diálogo modal con confirmación explícita para eliminar vectores en Qdrant."""
    if selected_video and selected_video in indexed_videos:
        st.markdown(f"**Eliminar vectores del vídeo activo:** `{selected_video}`")
        if st.button(
            f"Eliminar '{selected_video}' de la BD",
            icon=":material/delete:",
            key="dialog_delete_selected_btn",
            width="stretch",
        ):
            try:
                service.delete_video(selected_video)
                if st.session_state.get("active_video") == selected_video:
                    st.session_state["active_video"] = None
                    st.session_state["play_time"] = 0
                st.session_state["last_results"] = []
                st.session_state["pending_toast"] = (
                    f"Vectores de '{selected_video}' eliminados."
                )
                st.rerun()
            except (VectorDBError, OSError) as exc:
                st.error(f"Error al eliminar: {exc}", icon=":material/error:")

    st.warning(
        "Vaciar la colección elimina todos los vectores e índices del motor activo en Qdrant.",
        icon=":material/warning:",
        title="Acción irreversible",
    )
    confirm_clear = st.checkbox(
        "Confirmo que deseo vaciar toda la base de datos",
        key="dialog_confirm_clear_db",
    )
    if st.button(
        "Limpiar base de datos completa",
        icon=":material/delete_forever:",
        type="primary",
        disabled=not confirm_clear,
        key="dialog_clear_all_btn",
        width="stretch",
    ):
        try:
            service.clear_database()
            st.session_state["last_results"] = []
            st.session_state["active_video"] = None
            st.session_state["play_time"] = 0
            st.session_state["pending_toast"] = "Base de datos vectorial vaciada por completo."
            st.rerun()
        except (VectorDBError, OSError) as exc:
            st.error(f"Error al limpiar la BD: {exc}", icon=":material/error:")


def render_sidebar(
    service: VideoRAGService,
    auditor: RunAuditor,
) -> tuple[str | None, int, bool, bool]:
    """Renderiza la barra lateral con selección de vídeo, motor vectorial y búsqueda."""
    indexed_videos = _safe_indexed_videos(service)

    st.sidebar.subheader("Fuente de vídeo", icon=":material/video_library:")
    selected_video: str | None = None
    if indexed_videos:
        default_idx = 0
        current_active = st.session_state.get("active_video")
        if current_active in indexed_videos:
            default_idx = indexed_videos.index(current_active)

        selected_video = st.sidebar.selectbox(
            "Vídeo de trabajo",
            options=indexed_videos,
            index=default_idx,
            help="Vídeos indexados en Qdrant disponibles para consulta.",
        )
    else:
        st.sidebar.caption("Sin vídeos indexados en la colección activa.")

    ingest_clicked = st.sidebar.button(
        "Indexar o subir vídeo",
        icon=":material/upload_file:",
        width="stretch",
    )

    st.sidebar.subheader("Motor vectorial", icon=":material/memory:")
    prev_backend = service.embedding_backend
    prev_dim = service.vector_size
    backend_keys = list(EMBEDDING_BACKENDS)
    selected_backend = st.sidebar.segmented_control(
        "Motor de embeddings",
        options=backend_keys,
        default=prev_backend if prev_backend in backend_keys else backend_keys[0],
        required=True,
        key="embedding_backend",
        format_func=lambda key: BACKEND_UI_LABELS[key],
        width="stretch",
        wrap=True,
        help="Motor multimodal usado para vectorizar vídeos y consultas.",
    )
    p_backend = (
        selected_backend
        if isinstance(selected_backend, str) and selected_backend in EMBEDDING_BACKENDS
        else prev_backend
    )
    backend_spec = EMBEDDING_BACKENDS[p_backend]

    dim_options = list(mrl_options(p_backend))
    target_dim = (
        service.vector_size
        if p_backend == prev_backend
        else default_dimension(p_backend)
    )
    if target_dim not in dim_options:
        target_dim = dim_options[0]

    selected_dim = st.sidebar.segmented_control(
        "Dimensión Matryoshka (MRL)",
        options=dim_options,
        default=target_dim,
        required=True,
        key=f"mrl_dim_{p_backend}",
        format_func=lambda dim: f"{dim}d",
        width="stretch",
        help="Dimensiones soportadas por el motor activo con normalización L2.",
    )
    p_output_dimensionality = int(
        selected_dim if isinstance(selected_dim, int) and selected_dim in dim_options else target_dim
    )
    backend_caption_slot = st.sidebar.empty()

    st.sidebar.subheader("Recuperación", icon=":material/manage_search:")
    top_k = st.sidebar.slider(
        "Top K resultados",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Resultados finales mostrados. Con Etapa 2 activa, Qdrant recupera "
            "el doble de candidatos (2x) antes de reordenar."
        ),
    )
    enable_reranker = st.sidebar.toggle(
        "Verificación bimodal VLM (Etapa 2)",
        value=True,
        help="Verifica los fotogramas con Gemini 3.5 Flash Lite para descartar falsos positivos.",
    )
    filter_by_selected = False
    if selected_video:
        filter_by_selected = st.sidebar.toggle(
            f"Limitar búsqueda a '{selected_video}'",
            value=False,
            help="Si está desactivado, la búsqueda compara contra todos los vídeos indexados.",
        )

    with st.sidebar.expander(
        "Parámetros de indexación",
        expanded=False,
        icon=":material/tune:",
    ):
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

        st.markdown("**Vectorización y red**")
        p_max_concurrent = st.slider(
            "Concurrencia máxima API",
            min_value=1,
            max_value=10,
            value=1 if backend_spec.is_local else 3,
            key=f"max_concurrent_{p_backend}",
            disabled=backend_spec.is_local,
            help=(
                "Los modelos locales ejecutan 1 chunk cada vez para proteger la memoria unificada."
                if backend_spec.is_local
                else "Peticiones simultáneas a Gemini (3 para capa gratuita de 30K TPM)."
            ),
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
            help="Redimensiona los fotogramas clave antes de vectorizar.",
        )

        st.markdown("**Transcripción de audio (ASR)**")
        p_enable_audio = st.toggle(
            "Transcribir y alinear audio (Whisper)",
            value=True,
            help="Extrae la pista de voz con ffmpeg y la alinea temporalmente con cada plano.",
        )
        selected_whisper = st.segmented_control(
            "Modelo Whisper",
            options=["tiny", "base", "small"],
            default=WHISPER_MODEL,
            required=True,
            disabled=not p_enable_audio,
            width="stretch",
            help="Modelo local de OpenAI Whisper para reconocimiento de voz.",
        )
        p_whisper_model = (
            selected_whisper
            if isinstance(selected_whisper, str) and selected_whisper in {"tiny", "base", "small"}
            else WHISPER_MODEL
        )

    service.set_embedding(p_backend, p_output_dimensionality)
    caption_lines = [
        f"Modelo: `{backend_spec.model_id}` | Dispositivo: {device_label_for(p_backend)}"
    ]
    stats = service.last_index_stats
    if (
        stats is not None
        and stats.backend == p_backend
        and stats.dimension == p_output_dimensionality
    ):
        sec_per_chunk = stats.seconds / stats.chunks if stats.chunks > 0 else 0.0
        caption_lines.append(
            f"Última indexación: {stats.chunks} chunks en {stats.seconds:.1f} s "
            f"({sec_per_chunk:.2f} s/chunk)"
        )
    backend_caption_slot.caption("\n\n".join(caption_lines))

    # Sync sidebar video list immediately when switching backend or MRL dimension.
    if (p_backend, p_output_dimensionality) != (prev_backend, prev_dim):
        st.rerun()

    params = IndexingParams(
        threshold=p_threshold,
        min_shot_frames=p_min_shot_frames,
        entropy_stride=p_entropy_stride,
        output_dimensionality=p_output_dimensionality,
        max_concurrent=p_max_concurrent,
        jpeg_quality=p_jpeg_quality,
        target_width=p_target_width,
        enable_audio=p_enable_audio,
        whisper_model=p_whisper_model,
        embedding_backend=p_backend,
    )
    st.session_state["indexing_params"] = params
    st.session_state["indexed_videos"] = indexed_videos

    st.sidebar.subheader("Sistema y auditoría", icon=":material/shield:")
    maintenance_clicked = st.sidebar.button(
        "Mantenimiento de base de datos",
        icon=":material/storage:",
        width="stretch",
    )
    st.sidebar.caption(
        f"Carpeta activa: `audit/{auditor.run_dir.name}`\n\n"
        "Registra logs de ejecución, configuración y búsquedas en tiempo real."
    )

    if ingest_clicked:
        open_ingest_dialog(service, auditor, params, indexed_videos, selected_video)
    elif maintenance_clicked:
        open_maintenance_dialog(service, indexed_videos, selected_video)

    return selected_video, top_k, filter_by_selected, enable_reranker


def _init_session_state(selected_video: str | None) -> None:
    """Inicializa y sincroniza las claves de estado de la sesión UI."""
    if "play_time" not in st.session_state:
        st.session_state["play_time"] = 0
    if selected_video:
        if st.session_state.get("active_video") != selected_video:
            st.session_state["play_time"] = 0
        st.session_state["active_video"] = selected_video
    else:
        st.session_state["active_video"] = None
        st.session_state["last_results"] = []
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "last_audio_transcripts" not in st.session_state:
        st.session_state["last_audio_transcripts"] = []


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="TFM Video RAG Multimodal",
        page_icon=":material/movie_info:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pending_toast = st.session_state.pop("pending_toast", None)
    if pending_toast:
        st.toast(pending_toast, icon=":material/check_circle:")

    service = get_rag_service()
    auditor = get_auditor()
    selected_video, top_k, filter_by_selected, enable_reranker = render_sidebar(
        service, auditor
    )
    _init_session_state(selected_video)

    indexed_videos = st.session_state.get("indexed_videos") or _safe_indexed_videos(service)
    params: IndexingParams = st.session_state["indexing_params"]

    render_station_header(
        backend_label=BACKEND_UI_LABELS[service.embedding_backend],
        vector_size=service.vector_size,
        device_label=device_label_for(service.embedding_backend),
        indexed_count=len(indexed_videos),
    )

    col_left, col_right = st.columns([1.08, 0.92], gap="large")

    with col_left:
        active_video = st.session_state.get("active_video")
        play_time = int(st.session_state.get("play_time", 0))
        video_file_path = RAW_DIR / active_video if active_video else None

        render_video_monitor(
            active_video=active_video,
            play_time=play_time,
            video_file_path=video_file_path,
            on_open_ingest=lambda: open_ingest_dialog(
                service, auditor, params, indexed_videos, selected_video
            ),
            on_reindex=lambda path, name: _run_indexing_flow(
                video_path=path,
                video_name=name,
                status_label=f"Re-indexando {name}...",
                success_message=f"Re-indexados {{n}} chunks de '{name}'.",
                error_prefix="Error al re-indexar",
                service=service,
                auditor=auditor,
                params=params,
            ),
        )

    with col_right:
        st.subheader("Consulta multimodal", icon=":material/travel_explore:")
        active_video = st.session_state.get("active_video")
        search_disabled = active_video is None

        query_input = st.text_input(
            "Describe el evento, plano o diálogo que buscas",
            placeholder="Ej: un león rugiendo, bucles for, pingüinos en el agua...",
            help="Escribe en lenguaje natural o combina tu consulta con audio, imágenes y archivos .txt.",
            disabled=search_disabled,
        )

        with st.container(horizontal=True, vertical_alignment="center"):
            search_clicked = st.button(
                "Buscar evento",
                icon=":material/search:",
                type="primary",
                width="stretch",
                disabled=search_disabled,
            )

            with st.popover(
                "Audio",
                icon=":material/mic:",
                width="content",
                disabled=search_disabled,
            ):
                recorded_audio = st.audio_input(
                    "Grabar voz desde el micrófono",
                    key="query_mic_audio",
                )
                uploaded_audios = st.file_uploader(
                    "Subir pistas de audio (.wav, .mp3, .m4a, .ogg)",
                    type=["wav", "mp3", "m4a", "ogg"],
                    accept_multiple_files=True,
                    key="query_uploaded_audios",
                )

            with st.popover(
                "Imagen / .txt",
                icon=":material/add_photo_alternate:",
                width="content",
                disabled=search_disabled,
            ):
                uploaded_images = st.file_uploader(
                    "Subir imágenes de referencia (.png, .jpg, .jpeg, .webp)",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key="query_uploaded_images",
                )
                uploaded_txts = st.file_uploader(
                    "Subir archivos de texto (.txt)",
                    type=["txt"],
                    accept_multiple_files=True,
                    key="query_uploaded_txts",
                )

        ui_query = build_ui_multimodal_query(
            text_input=query_input,
            uploaded_txts=uploaded_txts,
            uploaded_images=uploaded_images,
            recorded_audio=recorded_audio,
            uploaded_audios=uploaded_audios,
        )

        if search_clicked:
            if not ui_query.has_content:
                st.warning(
                    "Introduce una consulta de texto o adjunta al menos una imagen, "
                    "un audio o un archivo .txt.",
                    icon=":material/warning:",
                )
            else:
                video_filter = active_video if filter_by_selected else None
                spinner_msg = (
                    "Etapa 1: Vectorización multimodal + Etapa 2: Verificación bimodal "
                    "con Gemini 3.5 Flash Lite..."
                    if enable_reranker
                    else "Generando embedding multimodal unificado y consultando Qdrant..."
                )
                with st.spinner(spinner_msg):
                    try:
                        results = asyncio.run(
                            execute_search(
                                query=ui_query,
                                service=service,
                                limit=top_k,
                                video_filter=video_filter,
                                auditor=auditor,
                                use_reranker=enable_reranker,
                            )
                        )
                        resolved = getattr(service, "last_resolved_query", None)
                        if isinstance(resolved, MultimodalQuery):
                            st.session_state["last_audio_transcripts"] = list(
                                resolved.audio_transcripts
                            )
                            display_label = (
                                resolved.composed_text
                                if resolved.composed_text
                                else f"[{len(resolved.images)} imagen(es) de referencia]"
                            )
                        else:
                            st.session_state["last_audio_transcripts"] = []
                            display_label = (
                                ui_query.composed_text
                                or f"[{len(ui_query.images)} imagen(es) de referencia]"
                            )
                        st.session_state["last_query"] = display_label
                        st.session_state["last_results"] = results
                    except ValueError as exc:
                        st.warning(f"Consulta no válida: {exc}", icon=":material/warning:")
                    except (VectorizationError, VectorDBError, OSError) as exc:
                        st.error(
                            f"Error al realizar la búsqueda: {exc}",
                            icon=":material/error:",
                        )

        render_attachment_preview(
            ui_query,
            st.session_state.get("last_audio_transcripts", []),
        )

        search_scope_label = (
            f"Solo '{active_video}'"
            if (filter_by_selected and active_video)
            else f"Todos los vídeos ({len(indexed_videos)})"
        )

        def _jump_to_timestamp(video_name: str, start_seconds: int) -> None:
            st.session_state["active_video"] = video_name
            st.session_state["play_time"] = start_seconds
            st.rerun()

        render_results_section(
            last_results=st.session_state.get("last_results", []),
            last_query=st.session_state.get("last_query", ""),
            top_k=top_k,
            search_scope_label=search_scope_label,
            search_clicked=search_clicked,
            on_jump=_jump_to_timestamp,
        )


if __name__ == "__main__":
    main()
