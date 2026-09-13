"""Aurion Local STT service — Parakeet provider, HTTP one-shot, V1.

Frontière figée par AURION_LOCAL_SPEECH_SERVICE_SPECIFICATION_AUDIT et son
arbitrage : ce service ne fait QUE décoder un segment audio déjà découpé
(par le VAD navigateur d'Aurion — jamais par ce service) et le transcrire
via Parakeet. Rien d'autre.

NE CONTIENT PAS, et ne doit jamais gagner :
- tmux, Claude Code, les fichiers ~/.claude/* ;
- Kokoro / ElevenLabs (aucun TTS ici) ;
- Qwen polish (le texte retourné est TOUJOURS le brut Parakeet) ;
- Silero VAD / SmartTurn EOU / pVAD (DEFER — le VAD reste côté navigateur
  Aurion pour ce lot) ;
- wake word ;
- toute décision de repli cloud (`OPENAI_STT` etc.) — un échec ici est une
  erreur HTTP typée ; c'est Aurion (SpeechService) qui décide d'un repli,
  jamais ce service.

Sécurité : lié à 127.0.0.1 uniquement (voir `run()`), et exige un jeton
partagé (`STT_SERVICE_TOKEN`) sur chaque requête — l'appelant est le
backend Aurion (SpeechService), jamais un navigateur directement, mais
« localhost » seul n'est pas une frontière de sécurité suffisante (un
onglet tiers pourrait tenter un appel direct) : le jeton est la vraie
barrière, l'adresse de liaison n'en est qu'une première.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request

from jarvis.transcriber import preload, transcribe
from services.audio_decode import AudioDecodeError, decode_to_pcm16

logger = logging.getLogger("aurion_local_stt")

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

#: Un énoncé vocal ne dépasse pas quelques Mo — au-delà, c'est refusé
#: plutôt que bufferisé indéfiniment (§19/§25 de la spec).
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

#: Un seul flux d'inférence à la fois — Parakeet/Metal ne sont pas
#: mesurés comme sûrs en accès concurrent (§18 de la spec).
_inference_lock = asyncio.Lock()


def _require_token(authorization: str | None) -> None:
    expected = os.environ.get("STT_SERVICE_TOKEN", "")
    if not expected:
        # Fail-closed : un service sans jeton configuré refuse TOUT,
        # il ne s'ouvre jamais par défaut.
        raise HTTPException(
            status_code=503,
            detail="service_not_configured: STT_SERVICE_TOKEN missing",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing_token")
    token = authorization[len("Bearer "):].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid_token")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        preload()
        app.state.model_loaded = True
        logger.info("Parakeet model preloaded: %s", MODEL_ID)
        yield

    app = FastAPI(title="Aurion Local STT (Parakeet)", lifespan=_lifespan)
    app.state.model_loaded = False

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": bool(app.state.model_loaded),
            "model": MODEL_ID,
        }

    @app.post("/v1/transcribe")
    async def transcribe_endpoint(
        request: Request, authorization: str | None = Header(default=None)
    ):
        _require_token(authorization)

        body = await request.body()
        if len(body) > MAX_PAYLOAD_BYTES:
            raise HTTPException(status_code=413, detail="payload_too_large")
        if not body:
            raise HTTPException(status_code=400, detail="empty_audio")

        loop = asyncio.get_running_loop()

        try:
            pcm = await loop.run_in_executor(None, decode_to_pcm16, body)
        except AudioDecodeError as exc:
            logger.warning("decode failed: %s", exc)
            raise HTTPException(
                status_code=422, detail=f"decode_error: {exc}"
            ) from None

        try:
            async with _inference_lock:
                text = await loop.run_in_executor(None, transcribe, pcm, 16000)
        except Exception as exc:  # noqa: BLE001 — frontière de service : jamais un 500 opaque
            logger.exception("transcription failed")
            raise HTTPException(
                status_code=502, detail=f"transcription_error: {exc}"
            ) from None

        return {
            "text": text,
            "provider": "parakeet",
            "model": MODEL_ID,
            "raw": True,
        }

    return app


def run(host: str = "127.0.0.1", port: int = 8100) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    run()
