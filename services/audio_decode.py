"""Decode arbitrary browser-recorded audio to PCM — via ffmpeg.

Aurion's frontend records with `MediaRecorder` (WebM/Opus, occasionally
WAV) — never raw PCM. `jarvis.transcriber.transcribe()` expects an already
decoded float32 mono array at a fixed sample rate. This module is the
missing link, measured as necessary by
AURION_LOCAL_SPEECH_SERVICE_SPECIFICATION_AUDIT.

ffmpeg is used as a universal decoder (WebM/Opus, WAV, MP4, OGG, ...)
rather than a codec-specific Python library — it already handles
resampling and channel downmixing correctly, and fails loudly and exactly
on corrupted/unsupported input instead of silently producing empty audio.

A decode failure is always an `AudioDecodeError` — it must never be
swallowed into a silent empty transcript. Whether the audio contains
speech is a question for Parakeet, not for this module.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

TARGET_SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    """Raised when audio bytes cannot be decoded to PCM."""


def decode_to_pcm16(audio_bytes: bytes) -> np.ndarray:
    """Decode `audio_bytes` (any ffmpeg-supported container/codec) to a
    mono float32 PCM array at `TARGET_SAMPLE_RATE`.

    Raises `AudioDecodeError` on empty input, missing ffmpeg, corrupted
    data, or an unsupported format — never returns an empty/garbage array
    silently."""
    if not audio_bytes:
        raise AudioDecodeError("empty_audio: no bytes to decode")

    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg_missing: decoder unavailable on this host")

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg_decode_failed: {stderr[:300]}")

    if not proc.stdout:
        raise AudioDecodeError("decode_produced_no_audio")

    return np.frombuffer(proc.stdout, dtype=np.float32)
