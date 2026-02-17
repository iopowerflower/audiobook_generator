"""
Gradio API Proxy for Kani TTS 2 – English Accents (kani-en)

Translates our REST API into Gradio calls against the HuggingFace Space.
Voice cloning supported via session-based embedding flow.
"""

import os
import io
import json
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from gradio_client import Client, handle_file

app = FastAPI(title="Kani TTS Proxy – English Accents")

GRADIO_CLIENT: Optional[Client] = None
GRADIO_URL = os.environ.get("GRADIO_URL", "http://localhost:7860")

SPEAKERS: list[str] = []
MODELS: list[str] = []

CLONED_VOICES: dict[str, bool] = {}

MIN_CLONE_SECONDS = 1.0
MIN_CLONE_SAMPLES_16K = int(MIN_CLONE_SECONDS * 16000)


async def get_gradio_client() -> Client:
    global GRADIO_CLIENT
    if GRADIO_CLIENT is None:
        GRADIO_CLIENT = Client(GRADIO_URL)
    return GRADIO_CLIENT


async def refresh_api_metadata(force: bool = False) -> None:
    global SPEAKERS, MODELS
    if not force and SPEAKERS and MODELS:
        return

    client = await get_gradio_client()
    api_schema = client.view_api(return_format="dict")
    speech_ep = (api_schema.get("named_endpoints") or {}).get(
        "/generate_speech_gpu", {}
    )
    params = speech_ep.get("parameters") or []

    new_models = []
    new_speakers = []
    if len(params) > 1:
        new_models = list((params[1].get("type") or {}).get("enum") or [])
    if len(params) > 3:
        new_speakers = list((params[3].get("type") or {}).get("enum") or [])

    MODELS = new_models or ["KaniTTS-2-en-accents"]
    SPEAKERS = new_speakers or ["Frank from Boston"]


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0
    language: str = "en"
    emotion: Optional[str] = None
    speaker_embedding: Optional[list[float]] = None
    custom_params: Optional[dict] = None


# ── Startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print(f"Connecting to Gradio backend at {GRADIO_URL}...")
    try:
        await refresh_api_metadata(force=True)
        print(f"Connected! Speakers: {len(SPEAKERS)}, Models: {len(MODELS)}")
    except Exception as e:
        print(f"Warning: Could not connect to Gradio backend: {e}")
        print("Will retry on first request...")


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        client = await get_gradio_client()
        return {
            "status": "ok",
            "gradio_url": GRADIO_URL,
            "speakers": len(SPEAKERS),
            "models": len(MODELS),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# ── Capabilities ─────────────────────────────────────────────────────

@app.get("/capabilities")
async def capabilities():
    try:
        await refresh_api_metadata()
    except Exception:
        pass
    return {
        "has_predefined_voices": True,
        "voice_cloning": True,
        "voice_cloning_formats": ["wav", "mp3", "ogg", "flac"],
        "emotion_tags": False,
        "emotion_options": [],
        "streaming": False,
        "sample_rate": 22050,
        "output_format": "wav",
        "multilingual": False,
        "languages": ["en"],
        "speaker_embedding": True,
        "embedding_dim": 128,
        "custom_params": {
            "temperature": {
                "type": "slider", "min": 0.1, "max": 1.5,
                "default": 1.0, "step": 0.05,
            },
            "top_p": {
                "type": "slider", "min": 0.1, "max": 1.0,
                "default": 0.95, "step": 0.05,
            },
            "repetition_penalty": {
                "type": "slider", "min": 1.0, "max": 2.0,
                "default": 1.1, "step": 0.05,
            },
            "model": {
                "type": "select",
                "options": MODELS,
                "default": MODELS[0] if MODELS else "KaniTTS-2-en-accents",
            },
        },
    }


# ── Voices ───────────────────────────────────────────────────────────

@app.get("/voices")
async def voices():
    try:
        await refresh_api_metadata()
    except Exception:
        pass

    voice_list = []
    for name in SPEAKERS:
        voice_list.append({
            "id": f"speaker:{name}",
            "name": name,
            "gender": "UNKNOWN",
            "language": "en",
            "description": "Predefined speaker",
            "is_cloned": False,
        })

    voice_list.append({
        "id": "random",
        "name": "Random Voice",
        "gender": "UNKNOWN",
        "language": "en",
        "description": "Model generates with random speaker characteristics",
        "is_cloned": False,
    })

    for cname in CLONED_VOICES:
        voice_list.append({
            "id": f"cloned:{cname}",
            "name": f"🎤 {cname}",
            "gender": "UNKNOWN",
            "language": "en",
            "description": "Cloned voice (session-level)",
            "is_cloned": True,
        })

    return {"voices": voice_list}


# ── Synthesize ───────────────────────────────────────────────────────

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    try:
        client = await get_gradio_client()
        await refresh_api_metadata()

        custom = request.custom_params or {}
        temperature = custom.get("temperature", 1.0)
        top_p = custom.get("top_p", 0.95)
        repetition_penalty = custom.get("repetition_penalty", 1.1)
        model_choice = custom.get(
            "model", MODELS[0] if MODELS else "KaniTTS-2-en-accents"
        )

        # Ensure we have speaker list (Gradio validates dropdown even in json mode)
        if not SPEAKERS:
            await refresh_api_metadata(force=True)
        default_speaker = SPEAKERS[0] if SPEAKERS else "Frank from Boston"

        if request.speaker_embedding:
            mode = "json"
            speaker_choice = default_speaker
            json_input = json.dumps(request.speaker_embedding)
        elif request.voice_id.startswith("cloned:"):
            # Cloned voices with embedding stored on Flask side come through
            # with speaker_embedding set; this branch is a fallback
            mode = "json"
            speaker_choice = default_speaker
            json_input = "[]"
        elif request.voice_id.startswith("speaker:"):
            mode = "select"
            speaker_choice = request.voice_id.replace("speaker:", "")
            json_input = ""
        else:
            mode = "select"
            speaker_choice = default_speaker
            json_input = ""

        result = client.predict(
            request.text,
            model_choice,
            mode,
            speaker_choice,
            json_input,
            temperature,
            top_p,
            repetition_penalty,
            api_name="/generate_speech_gpu",
        )

        if isinstance(result, tuple):
            sample_rate, audio = result
        elif isinstance(result, str):
            audio, sample_rate = sf.read(result)
        elif isinstance(result, dict) and result.get("path"):
            audio, sample_rate = sf.read(result["path"])
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        audio_bytes = buf.getvalue()
        duration = len(audio) / sample_rate

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Audio-Format": "wav",
                "X-Duration-Seconds": str(duration),
            },
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Voice Cloning ────────────────────────────────────────────────────

@app.post("/clone_voice")
async def clone_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    sample_rate: int = Form(16000),
):
    """
    Clone a voice from an uploaded audio sample.

    The embedding is stored in the Gradio session State.
    Future calls with mode="generate" will use this embedding.
    """
    try:
        client = await get_gradio_client()

        raw_bytes = await audio.read()
        data, sr = sf.read(io.BytesIO(raw_bytes))

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != 16000:
            target_len = int(len(data) * 16000 / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
            sr = 16000

        if len(data) < MIN_CLONE_SAMPLES_16K:
            pad_len = MIN_CLONE_SAMPLES_16K - len(data)
            data = np.concatenate([data, np.zeros(pad_len, dtype=data.dtype)])

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, data, sr)
            result = client.predict(
                handle_file(tmp.name),
                api_name="/generate_embedding_gpu",
            )
        finally:
            os.unlink(tmp.name)

        if "error" in str(result).lower() and "ready" not in str(result).lower():
            raise HTTPException(status_code=422, detail=f"Embedding failed: {result}")

        CLONED_VOICES[name] = True

        return {
            "id": f"cloned:{name}",
            "name": name,
            "language": "en",
            "is_cloned": True,
            "message": str(result),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7862))
    uvicorn.run(app, host="0.0.0.0", port=port)
