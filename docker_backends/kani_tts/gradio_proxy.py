"""
Gradio API Proxy for Kani TTS

This provides a REST API that calls the Gradio interface of the HuggingFace Space.
Much simpler than rebuilding the entire model stack - just use what already works.

The HF Space exposes a Gradio API that we can call programmatically.
"""

import os
import io
import json
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from gradio_client import Client

app = FastAPI(title="Kani TTS Proxy")

# Gradio client (connects to the HF Space)
GRADIO_CLIENT: Optional[Client] = None
GRADIO_URL = os.environ.get("GRADIO_URL", "http://localhost:7860")

# Cache available speakers and models
SPEAKERS = []
MODELS = []


async def refresh_api_metadata(force: bool = False) -> None:
    """
    Refresh model/speaker metadata from the Space API schema.
    Safe to call repeatedly; only refreshes when needed unless force=True.
    """
    global SPEAKERS, MODELS
    if not force and SPEAKERS and MODELS:
        return

    client = await get_gradio_client()
    api_schema = client.view_api(return_format="dict")
    speech_ep = (api_schema.get("named_endpoints") or {}).get("/generate_speech_gpu", {})
    params = speech_ep.get("parameters") or []

    # Param index 1 = model_choice, index 3 = speaker_choice in this space.
    new_models = []
    new_speakers = []
    if len(params) > 1:
        new_models = list((params[1].get("type") or {}).get("enum") or [])
    if len(params) > 3:
        new_speakers = list((params[3].get("type") or {}).get("enum") or [])

    # Safe fallbacks
    MODELS = new_models or ["KaniTTS-2-pt"]
    SPEAKERS = new_speakers or ["Robert (en)"]


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0
    language: str = "en"
    emotion: Optional[str] = None
    speaker_embedding: Optional[list[float]] = None
    custom_params: Optional[dict] = None


async def get_gradio_client() -> Client:
    """Get or create Gradio client."""
    global GRADIO_CLIENT
    if GRADIO_CLIENT is None:
        GRADIO_CLIENT = Client(GRADIO_URL)
    return GRADIO_CLIENT


@app.on_event("startup")
async def startup():
    """Initialize connection to Gradio backend."""
    global SPEAKERS, MODELS
    
    print(f"Connecting to Gradio backend at {GRADIO_URL}...")
    
    try:
        await refresh_api_metadata(force=True)
        
        print(f"Connected! Speakers: {len(SPEAKERS)}, Models: {len(MODELS)}")
    except Exception as e:
        print(f"Warning: Could not connect to Gradio backend: {e}")
        print("Will retry on first request...")


@app.get("/health")
async def health():
    """Health check."""
    try:
        client = await get_gradio_client()
        # Quick test - just check if we can reach the API
        return {
            "status": "ok",
            "gradio_url": GRADIO_URL,
            "speakers": len(SPEAKERS),
            "models": len(MODELS),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/capabilities")
async def capabilities():
    """Return backend capabilities."""
    try:
        await refresh_api_metadata()
    except Exception:
        pass
    return {
        "has_predefined_voices": True,
        # This proxy cannot retrieve embedding vectors from the Space API
        # because /generate_embedding_gpu only returns status text.
        "voice_cloning": False,
        "voice_cloning_formats": [],
        "emotion_tags": False,
        "emotion_options": [],
        "streaming": False,
        "sample_rate": 22050,
        "output_format": "wav",
        "multilingual": True,
        "languages": ["en", "ky", "es"],
        "speaker_embedding": True,
        "embedding_dim": 128,
        "custom_params": {
            "temperature": {"type": "slider", "min": 0.1, "max": 1.5, "default": 1.0, "step": 0.05},
            "top_p": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.95, "step": 0.05},
            "repetition_penalty": {"type": "slider", "min": 1.0, "max": 2.0, "default": 1.1, "step": 0.05},
            "model": {"type": "select", "options": MODELS, "default": MODELS[0] if MODELS else "kani-2-pt-450m"},
        },
    }


@app.get("/voices")
async def voices():
    """Return available voices."""
    try:
        await refresh_api_metadata()
    except Exception:
        pass
    voice_list = []
    
    for name in SPEAKERS:
        lang = "en"
        if "(" in name:
            lang = name.split("(")[-1].replace(")", "").strip()
        
        voice_list.append({
            "id": f"speaker:{name}",
            "name": name,
            "gender": "UNKNOWN",
            "language": lang,
            "description": f"Predefined speaker",
            "is_cloned": False,
        })
    
    # Add "random" voice option
    voice_list.append({
        "id": "random",
        "name": "Random Voice",
        "gender": "UNKNOWN",
        "language": "en",
        "description": "Model will generate with random speaker characteristics",
        "is_cloned": False,
    })
    
    return {"voices": voice_list}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech via Gradio API."""
    try:
        client = await get_gradio_client()
        await refresh_api_metadata()
        
        # Parse parameters
        custom = request.custom_params or {}
        temperature = custom.get("temperature", 1.0)
        top_p = custom.get("top_p", 0.95)
        repetition_penalty = custom.get("repetition_penalty", 1.1)
        model_choice = custom.get("model", MODELS[0] if MODELS else "kani-2-pt-450m")
        
        # Determine speaker mode and value
        if request.speaker_embedding:
            # Clone mode with embedding
            mode = "json"
            speaker_choice = SPEAKERS[0] if SPEAKERS else "Kore (en)"
            json_input = json.dumps(request.speaker_embedding)
        elif request.voice_id.startswith("speaker:"):
            # Select predefined speaker
            mode = "select"
            speaker_choice = request.voice_id.replace("speaker:", "")
            json_input = ""
        else:
            # Random/default
            mode = "select"
            speaker_choice = SPEAKERS[0] if SPEAKERS else "Kore (en)"
            json_input = ""
        
        # Call the Gradio API by name (stable across UI graph changes).
        # Signature: generate_speech_gpu(text, model_choice, mode, speaker_choice, json_input, t, top_p, rp)
        result = client.predict(
            request.text,           # text
            model_choice,           # model_choice
            mode,                   # mode: "select", "generate", or "json"
            speaker_choice,         # speaker_choice
            json_input,             # json_input
            temperature,            # t
            top_p,                  # top_p
            repetition_penalty,     # rp
            api_name="/generate_speech_gpu",
        )
        
        # Result can be filepath or FileData-like dict.
        if isinstance(result, tuple):
            sample_rate, audio = result
        elif isinstance(result, str):
            audio, sample_rate = sf.read(result)
        elif isinstance(result, dict) and result.get("path"):
            audio, sample_rate = sf.read(result["path"])
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        # Convert to WAV bytes
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
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clone_voice")
async def clone_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    sample_rate: int = Form(16000),
):
    """Voice cloning is not available via the Space's public API contract."""
    raise HTTPException(
        status_code=501,
        detail=(
            "Voice cloning is not supported in proxy mode. "
            "The Space endpoint /generate_embedding_gpu does not return embeddings."
        ),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7862))
    uvicorn.run(app, host="0.0.0.0", port=port)
