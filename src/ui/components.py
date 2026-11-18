"""Componentes visuales reutilizables para la interfaz Streamlit."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from src.models import MultimodalQuery, RerankedResult, SearchResult
from src.rag_engine.service import format_timestamp


@dataclass(frozen=True)
class IndexingParams:
    """Parámetros activos de segmentación, vectorización y transcripción."""

    threshold: float
    min_shot_frames: int
    entropy_stride: int
    output_dimensionality: int
    max_concurrent: int
    jpeg_quality: int
    target_width: int
    enable_audio: bool
    whisper_model: str
    embedding_backend: str


def render_station_header(
    backend_label: str,
    vector_size: int,
    device_label: str,
    indexed_count: int,
) -> None:
    """Renderiza el título principal y la barra de telemetría activa."""
    st.title("Video RAG Multimodal", icon=":material/movie_info:")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(backend_label, icon=":material/memory:", color="blue")
        st.badge(f"MRL {vector_size}d", icon=":material/straighten:", color="violet")
        st.badge(device_label, icon=":material/developer_board:", color="gray")
        count_label = (
            f"{indexed_count} vídeo en Qdrant"
            if indexed_count == 1
            else f"{indexed_count} vídeos en Qdrant"
        )
        st.badge(
            count_label,
            icon=":material/database:",
            color="green" if indexed_count > 0 else "orange",
        )


def render_video_monitor(
    active_video: str | None,
    play_time: int,
    video_file_path: Path | None,
    on_open_ingest: Callable[[], None],
    on_reindex: Callable[[Path, str], None],
) -> None:
    """Renderiza el monitor de reproducción o el estado vacío accionable."""
    st.subheader("Monitor de inspección", icon=":material/smart_display:")

    if not active_video:
        with st.container(border=True):
            st.info(
                "No hay vídeos indexados en la colección activa. Indexa un archivo "
                "de data/raw/ o sube uno nuevo para iniciar las consultas.",
                icon=":material/video_library:",
                title="Colección vacía",
            )
            if st.button(
                "Indexar primer vídeo",
                icon=":material/upload_file:",
                type="primary",
                key="empty_state_ingest_btn",
            ):
                on_open_ingest()
        return

    if video_file_path is None or not video_file_path.is_file():
        st.warning(
            f"El archivo `{active_video}` no se encuentra en `data/raw/`.",
            icon=":material/warning:",
            title="Archivo fuente no encontrado",
        )
        return

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(active_video, icon=":material/movie:", color="gray")
            st.markdown(f"Posición: `{format_timestamp(play_time)}` ({play_time}s)")

        st.video(
            str(video_file_path),
            start_time=play_time,
            alt=f"Monitor de reproducción para {active_video}",
        )

        with st.container(horizontal=True, vertical_alignment="center"):
            if st.button(
                "Re-indexar vídeo",
                icon=":material/restart_alt:",
                key="monitor_reindex_btn",
                width="content",
            ):
                on_reindex(video_file_path, active_video)


def render_attachment_preview(
    query: MultimodalQuery,
    audio_transcripts: Sequence[str],
) -> None:
    """Muestra una vista previa compacta de las modalidades adjuntas a la consulta."""
    has_attachments = bool(query.images or query.audio_bytes or query.txt_contents)
    if not has_attachments:
        return

    with st.container(border=True):
        st.caption("Adjuntos multimodales activos en la consulta")
        if query.images:
            img_cols = st.columns(min(len(query.images), 4))
            for idx, img_bytes in enumerate(query.images):
                img_cols[idx % len(img_cols)].image(
                    img_bytes,
                    caption=f"Imagen {idx + 1}",
                    alt=f"Imagen de referencia {idx + 1}",
                    width=110,
                )
        if query.audio_bytes:
            for idx, aud_bytes in enumerate(query.audio_bytes):
                st.audio(
                    aud_bytes,
                    alt=f"Pista de audio adjunta {idx + 1}",
                )
            if audio_transcripts:
                joined_tr = " ".join(audio_transcripts)
                st.caption(f'Transcripción detectada: "{joined_tr}"')
        if query.txt_contents:
            excerpt = " ".join(query.txt_contents)[:160]
            st.caption(f'Archivos .txt ({len(query.txt_contents)}): "{excerpt}"')


def render_result_card(
    rank: int,
    result: SearchResult | RerankedResult,
    on_jump: Callable[[str, int], None],
) -> None:
    """Renderiza una tarjeta de hallazgo sincronizada por código de tiempo."""
    start_tc = format_timestamp(result.start_time)
    end_tc = format_timestamp(result.end_time)

    with st.container(border=True):
        c1, c2 = st.columns([0.68, 0.32], vertical_alignment="center")

        with c1:
            st.markdown(
                f"**#{rank}** · `{start_tc} - {end_tc}` "
                f"(`{result.start_time:.1f}s - {result.end_time:.1f}s`) · `{result.video_name}`"
            )

        with c2:
            btn_key = f"play_btn_{result.video_name}_{result.chunk_index}_{rank}"
            if st.button(
                f"Ir a {start_tc}",
                key=btn_key,
                icon=":material/play_arrow:",
                width="stretch",
            ):
                on_jump(result.video_name, int(result.start_time))

        with st.container(horizontal=True, vertical_alignment="center"):
            if isinstance(result, RerankedResult):
                pct = max(0, min(100, int(result.relevance_score * 100)))
                if result.is_relevant:
                    st.badge(
                        f"Validado VLM ({pct}%)",
                        icon=":material/verified:",
                        color="green",
                    )
                else:
                    st.badge(
                        f"Descartado VLM ({pct}%)",
                        icon=":material/cancel:",
                        color="red",
                    )
                st.badge(
                    f"Coseno Qdrant: {result.vector_score:.4f}",
                    icon=":material/scatter_plot:",
                    color="blue",
                )
            else:
                st.badge(
                    f"Similitud coseno: {result.score:.4f}",
                    icon=":material/scatter_plot:",
                    color="blue",
                )

        if isinstance(result, RerankedResult) and result.reasoning:
            st.caption(f":material/psychology: **Verificación bimodal:** {result.reasoning}")

        if result.transcript:
            st.caption(f':material/graphic_eq: **Transcripción ASR:** "{result.transcript}"')


def render_results_section(
    last_results: Sequence[Any],
    last_query: str,
    top_k: int,
    search_scope_label: str,
    search_clicked: bool,
    on_jump: Callable[[str, int], None],
) -> None:
    """Renderiza el listado de hallazgos temporales o el mensaje de estado vacío."""
    if last_results:
        st.subheader("Hallazgos por código de tiempo", icon=":material/schedule:")
        st.caption(f'Consulta: "{last_query}" · Alcance: {search_scope_label}')
        for rank, result in enumerate(last_results[:top_k], start=1):
            render_result_card(rank, result, on_jump)
    elif last_query and not search_clicked:
        st.info(
            "No se encontraron fragmentos similares para esta consulta.",
            icon=":material/search_off:",
        )
