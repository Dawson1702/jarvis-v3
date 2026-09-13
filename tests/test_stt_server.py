"""Aurion Local STT service — HTTP contract tests.

Frontière figée par AURION_LOCAL_SPEECH_SERVICE_SPECIFICATION_AUDIT (V1) :
- one-shot HTTP, aucun WebSocket, aucun VAD serveur (Silero DEFER) ;
- reçoit un segment déjà découpé par le VAD navigateur d'Aurion ;
- décode (WebM/Opus/WAV/...) puis transcrit via Parakeet, texte BRUT,
  jamais de polish ;
- une panne de décodage ou de transcription est une erreur HTTP typée,
  JAMAIS un `200 {"text": ""}` silencieux ;
- le service ne décide jamais d'un repli cloud — ça reste une décision
  Aurion (SpeechService), hors de ce service.

La plupart des tests mockent `jarvis.transcriber.transcribe` (modèle MLX
lourd à charger) pour rester rapides et déterministes ; un seul test réel
de bout en bout (T11) charge le vrai modèle Parakeet pour prouver la
tranche verticale.
"""
from __future__ import annotations

import io
import subprocess

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from services import stt_server

TOKEN = "test-token-123"


def _wav_bytes(duration_s: float = 1.0, sr: int = 16000, channels: int = 1) -> bytes:
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    if channels == 2:
        tone = np.stack([tone, tone], axis=1)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _wav_to_webm(wav_bytes: bytes) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "webm", "-c:a", "libopus", "pipe:1"],
        input=wav_bytes, capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("STT_SERVICE_TOKEN", TOKEN)
    app = stt_server.create_app()
    with TestClient(app) as c:
        yield c


def _auth(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ═══ Health ═══

def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] == stt_server.MODEL_ID


# ═══ Auth (token local — pas seulement Origin) ═══

def test_transcribe_requires_token(client):
    r = client.post("/v1/transcribe", content=_wav_bytes())
    assert r.status_code == 401


def test_transcribe_rejects_wrong_token(client):
    r = client.post("/v1/transcribe", content=_wav_bytes(), headers=_auth("wrong-token"))
    assert r.status_code == 401


def test_service_refuses_all_requests_if_token_not_configured(monkeypatch):
    monkeypatch.delenv("STT_SERVICE_TOKEN", raising=False)
    app = stt_server.create_app()
    with TestClient(app) as c:
        r = c.post("/v1/transcribe", content=_wav_bytes(), headers=_auth("anything"))
        assert r.status_code == 503


# ═══ D3 / D4 / D5 — jamais un 200 silencieux sur panne de décodage ═══

def test_empty_audio_is_refused_not_silently_ok(client):
    r = client.post("/v1/transcribe", content=b"", headers=_auth())
    assert r.status_code == 400
    assert r.json()["detail"] == "empty_audio"


def test_corrupted_audio_is_a_typed_decode_error_not_200(client):
    garbage = b"\x00\x01\xffnot audio at all" * 50
    r = client.post("/v1/transcribe", content=garbage, headers=_auth())
    assert r.status_code == 422
    assert "decode_error" in r.json()["detail"]


def test_unsupported_format_is_a_typed_decode_error(client):
    # Texte brut : ffmpeg le refusera comme il refuserait n'importe quel
    # conteneur non reconnu.
    r = client.post("/v1/transcribe", content=b"this is plain text, not audio",
                     headers=_auth())
    assert r.status_code == 422


# ═══ D9 (implicite) — payload trop gros ═══

def test_payload_too_large_is_rejected(client, monkeypatch):
    monkeypatch.setattr(stt_server, "MAX_PAYLOAD_BYTES", 100)
    r = client.post("/v1/transcribe", content=_wav_bytes(duration_s=2.0), headers=_auth())
    assert r.status_code == 413


# ═══ D1 / D2 / D6 / D7 — décodage réel, transcription mockée ═══

def test_d1_valid_wav_is_decoded_and_transcribed(client, monkeypatch):
    monkeypatch.setattr(stt_server, "transcribe", lambda audio, sample_rate: "bonjour le monde")
    r = client.post("/v1/transcribe", content=_wav_bytes(), headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "text": "bonjour le monde",
        "provider": "parakeet",
        "model": stt_server.MODEL_ID,
        "raw": True,
    }


def test_d2_valid_webm_opus_is_decoded_and_transcribed(client, monkeypatch):
    monkeypatch.setattr(stt_server, "transcribe", lambda audio, sample_rate: "ouvre belfort")
    webm = _wav_to_webm(_wav_bytes())
    r = client.post("/v1/transcribe", content=webm, headers=_auth())
    assert r.status_code == 200
    assert r.json()["text"] == "ouvre belfort"


def test_d6_stereo_audio_is_downmixed_correctly(client, monkeypatch):
    captured = {}

    def _fake_transcribe(audio, sample_rate):
        captured["audio"] = audio
        captured["sample_rate"] = sample_rate
        return "ok"

    monkeypatch.setattr(stt_server, "transcribe", _fake_transcribe)
    r = client.post("/v1/transcribe", content=_wav_bytes(channels=2), headers=_auth())
    assert r.status_code == 200
    assert captured["sample_rate"] == 16000
    assert captured["audio"].ndim == 1  # mono après décodage


def test_d7_different_sample_rate_is_resampled_to_16k(client, monkeypatch):
    captured = {}

    def _fake_transcribe(audio, sample_rate):
        captured["audio"] = audio
        captured["sample_rate"] = sample_rate
        return "ok"

    monkeypatch.setattr(stt_server, "transcribe", _fake_transcribe)
    duration_s = 1.0
    r = client.post("/v1/transcribe", content=_wav_bytes(duration_s=duration_s, sr=48000),
                     headers=_auth())
    assert r.status_code == 200
    assert captured["sample_rate"] == 16000
    # ~16000 echantillons pour 1s a 16kHz (tolerance decodeur/resampler)
    assert abs(len(captured["audio"]) - 16000) < 200


# ═══ D8 — segment tres court : succes, transcript vide legitime ═══

def test_d8_very_short_segment_succeeds_with_legitimate_empty_transcript(client, monkeypatch):
    monkeypatch.setattr(stt_server, "transcribe", lambda audio, sample_rate: "")
    r = client.post("/v1/transcribe", content=_wav_bytes(duration_s=0.05), headers=_auth())
    assert r.status_code == 200
    assert r.json()["text"] == ""


# ═══ Raw transcript authority — jamais reformule ═══

def test_raw_transcript_is_never_altered_by_the_service(client, monkeypatch):
    exact = "  Ne lance PAS Belfort — deux cents euros.  "
    monkeypatch.setattr(stt_server, "transcribe", lambda audio, sample_rate: exact)
    r = client.post("/v1/transcribe", content=_wav_bytes(), headers=_auth())
    assert r.json()["text"] == exact


# ═══ T11 — tranche verticale reelle, aucun mock ═══

@pytest.mark.slow
def test_real_end_to_end_transcription_no_mock(client):
    """Le seul test qui charge le vrai modele Parakeet — preuve que le
    mecanisme reel fonctionne, pas seulement le contrat HTTP autour."""
    r = client.post("/v1/transcribe", content=_wav_bytes(duration_s=1.0), headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "parakeet"
    assert body["raw"] is True
    assert isinstance(body["text"], str)
