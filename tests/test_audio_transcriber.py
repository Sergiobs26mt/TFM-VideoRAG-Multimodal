"""Pruebas unitarias de ``src.processing.audio_transcriber`` (sin GPU ni red)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.models import AudioTranscriptionError, TranscriptSegment, VideoChunk
from src.processing import audio_transcriber
from src.processing.audio_transcriber import (
    align_transcript_with_chunks,
    extract_audio_track,
    transcribe_audio,
    transcribe_query_audios,
    transcribe_video_audio,
)


def test_extract_audio_track_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio_track(tmp_path / "no_existe.mp4")


def test_extract_audio_track_runs_ffmpeg_and_returns_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-video")
    out_wav = tmp_path / "out.wav"
    recorded_cmd: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmd.extend(cmd)
        assert kwargs.get("shell") is not True
        Path(cmd[-1]).write_bytes(b"RIFF-fake-wav-data")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = extract_audio_track(video, output_wav=out_wav)
    assert result == out_wav
    assert out_wav.is_file()
    assert "-vn" in recorded_cmd
    assert "16000" in recorded_cmd
    assert "pcm_s16le" in recorded_cmd


def test_extract_audio_track_returns_none_and_cleans_up_when_no_audio_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "silent_video.mp4"
    video.write_bytes(b"fake-video")
    created_paths: list[Path] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        wav_target = Path(cmd[-1])
        created_paths.append(wav_target)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="Output file does not contain any stream",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = extract_audio_track(video)
    assert result is None
    assert len(created_paths) == 1
    assert not created_paths[0].exists()


def test_extract_audio_track_raises_when_ffmpeg_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-video")
    out_wav = tmp_path / "temp.wav"

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("ffmpeg no encontrado")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AudioTranscriptionError, match="ffmpeg"):
        extract_audio_track(video, output_wav=out_wav)
    assert not out_wav.exists()


def test_transcribe_audio_with_injected_fn_filters_empty_and_sorts(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF-data")

    def fake_transcriber(
        path: str, model_name: str, language: str | None
    ) -> list[dict[str, Any]]:
        assert path == str(wav)
        assert model_name == "base"
        assert language == "es"
        return [
            {"start": 5.0, "end": 8.0, "text": "  segunda frase  "},
            {"start": 0.0, "end": 1.0, "text": "   "},
            {"start": 1.0, "end": 4.0, "text": "primera frase"},
        ]

    segments = transcribe_audio(wav, transcriber_fn=fake_transcriber)
    assert segments == [
        TranscriptSegment(start_time=1.0, end_time=4.0, text="primera frase"),
        TranscriptSegment(start_time=5.0, end_time=8.0, text="segunda frase"),
    ]


def test_transcribe_audio_caches_whisper_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF-data")

    load_calls: list[str] = []

    class _FakeModel:
        def transcribe(
            self, audio_path: str, *, language: str | None = None, fp16: bool = False
        ) -> dict[str, Any]:
            assert fp16 is False
            assert language == "es"
            return {
                "segments": [
                    {"start": 0.5, "end": 2.5, "text": " Diccionarios en Python "},
                ]
            }

    def fake_load_model(name: str) -> _FakeModel:
        load_calls.append(name)
        return _FakeModel()

    monkeypatch.setattr(audio_transcriber, "_MODEL_CACHE", {})
    monkeypatch.setattr(
        audio_transcriber,
        "whisper",
        SimpleNamespace(load_model=fake_load_model),
    )

    seg1 = transcribe_audio(wav, model_name="base", language="es")
    seg2 = transcribe_audio(wav, model_name="base", language="es")

    assert load_calls == ["base"]
    assert seg1 == seg2 == [
        TranscriptSegment(start_time=0.5, end_time=2.5, text="Diccionarios en Python")
    ]


def test_transcribe_video_audio_cleans_temporary_wav_even_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake-mp4")
    temp_wav = tmp_path / "extracted.wav"

    def fake_extract(
        video_path: str | Path, output_wav: Path | None = None
    ) -> Path | None:
        temp_wav.write_bytes(b"RIFF-data")
        return temp_wav

    monkeypatch.setattr(audio_transcriber, "extract_audio_track", fake_extract)

    def failing_transcriber(
        path: str, model_name: str, language: str | None
    ) -> list[dict[str, Any]]:
        assert Path(path).exists()
        raise AudioTranscriptionError("fallo ASR")

    with pytest.raises(AudioTranscriptionError, match="fallo ASR"):
        transcribe_video_audio(video, transcriber_fn=failing_transcriber)

    assert not temp_wav.exists()


def test_transcribe_video_audio_returns_empty_list_when_video_has_no_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "mute.mp4"
    video.write_bytes(b"fake-mp4")
    monkeypatch.setattr(audio_transcriber, "extract_audio_track", lambda *_, **__: None)

    assert transcribe_video_audio(video) == []


def test_align_transcript_with_chunks_handles_containment_boundary_overlap_and_silence() -> None:
    chunks = [
        VideoChunk(index=0, start_frame=0, end_frame=49, start_time=0.0, end_time=5.0),
        VideoChunk(index=1, start_frame=50, end_frame=99, start_time=5.0, end_time=10.0),
        VideoChunk(index=2, start_frame=100, end_frame=149, start_time=10.0, end_time=15.0),
    ]
    segments = [
        # Corto (0.1s < min_overlap=0.2) pero 100% contenido en el chunk 0
        TranscriptSegment(start_time=1.0, end_time=1.1, text="Hola."),
        # Cruza la frontera 5.0s con >= 0.2s en el chunk 0 (0.5s) y en el chunk 1 (0.6s)
        TranscriptSegment(
            start_time=4.5,
            end_time=5.6,
            text="Hoy veremos diccionarios.",
        ),
        # Solapamiento marginal (0.05s < 0.2s) con el chunk 1; pertenece solo al chunk 0
        TranscriptSegment(
            start_time=3.0,
            end_time=5.05,
            text="Preparando el editor.",
        ),
    ]

    aligned = align_transcript_with_chunks(chunks, segments, min_overlap=0.2)

    assert len(aligned) == 3
    assert aligned[0].transcript == (
        "Hola. Preparando el editor. Hoy veremos diccionarios."
    )
    assert aligned[1].transcript == "Hoy veremos diccionarios."
    assert aligned[2].transcript is None


def test_transcribe_query_audios_supports_paths_and_bytes_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "voice.mp3"
    audio_file.write_bytes(b"ID3-fake-mp3")
    created_wavs: list[Path] = []

    def fake_extract(
        source_path: str | Path, output_wav: Path | None = None
    ) -> Path | None:
        assert Path(source_path).is_file()
        wav = tmp_path / f"conv_{len(created_wavs)}.wav"
        wav.write_bytes(b"RIFF-16k")
        created_wavs.append(wav)
        return wav

    monkeypatch.setattr(audio_transcriber, "extract_audio_track", fake_extract)

    calls: list[str] = []

    def fake_transcriber(
        path: str, model_name: str, language: str | None
    ) -> list[dict[str, Any]]:
        calls.append(path)
        if len(calls) == 1:
            return [
                {"start": 0.0, "end": 1.5, "text": "busca un"},
                {"start": 1.5, "end": 3.0, "text": "camaleón verde"},
            ]
        return [{"start": 0.0, "end": 2.0, "text": "en la rama"}]

    transcripts = transcribe_query_audios(
        [audio_file, b"RIFF-mic-recording"],
        transcriber_fn=fake_transcriber,
    )
    assert transcripts == ["busca un camaleón verde", "en la rama"]
    assert len(created_wavs) == 2
    for wav in created_wavs:
        assert not wav.exists()


def test_transcribe_query_audios_degrades_gracefully_on_silence_or_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid_audio = tmp_path / "ok.wav"
    valid_audio.write_bytes(b"RIFF-ok")
    silent_audio = tmp_path / "silent.ogg"
    silent_audio.write_bytes(b"OggS-silent")
    missing_audio = tmp_path / "missing.m4a"

    def fake_extract(
        source_path: str | Path, output_wav: Path | None = None
    ) -> Path | None:
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(str(src))
        if src.name == "silent.ogg":
            return None
        wav = tmp_path / f"wav_{src.stem}.wav"
        wav.write_bytes(b"RIFF-16k")
        return wav

    monkeypatch.setattr(audio_transcriber, "extract_audio_track", fake_extract)

    def fake_transcriber(
        path: str, model_name: str, language: str | None
    ) -> list[dict[str, Any]]:
        if "ok" in path:
            return [{"start": 0.0, "end": 1.0, "text": "bucles for"}]
        raise AudioTranscriptionError("corrupto")

    result = transcribe_query_audios(
        [missing_audio, silent_audio, b"", b"corrupt-bytes", valid_audio],
        transcriber_fn=fake_transcriber,
    )
    assert result == ["bucles for"]


def test_unload_whisper_models_empties_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_transcriber, "_MODEL_CACHE", {"base": object(), "tiny": object()})
    audio_transcriber.unload_whisper_models()
    assert audio_transcriber._MODEL_CACHE == {}

