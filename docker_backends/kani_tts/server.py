"""
Kani TTS REST API Server

This runs inside a Docker container and exposes the standard TTS backend API.
Based on https://huggingface.co/spaces/nineninesix/kani-tts-2-pt
"""

import io
import os
import json
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Kani TTS Backend")

# Global model instances (loaded on startup)
MODELS = {}
SPEAKER_MANAGER = None
DEVICE = "cpu"


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0
    language: str = "en"
    emotion: Optional[str] = None
    speaker_embedding: Optional[list[float]] = None
    custom_params: Optional[dict] = None


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    global MODELS, SPEAKER_MANAGER, DEVICE
    
    # Import Kani TTS (installed from the HF space)
    from kani_tts import KaniTTS
    from util import SpeakerManager
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load models based on environment or defaults
    model_configs = {
        "kani-2-pt-450m": "nineninesix/kani-tts-2-pt-450m",
    }
    
    # Check for custom model config from environment
    custom_models = os.environ.get("KANI_MODELS")
    if custom_models:
        model_configs = json.loads(custom_models)
    
    for name, model_id in model_configs.items():
        print(f"Loading {name} from {model_id}...")
        try:
            MODELS[name] = KaniTTS(
                model_name=model_id,
                device_map=DEVICE,
            )
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    # Load speaker manager (handles predefined speakers + embedding generation)
    speaker_map_path = os.environ.get("SPEAKER_MAP", "./speakers/speaker_map.json")
    SPEAKER_MANAGER = SpeakerManager(speaker_map_path)
    print(f"Loaded {len(SPEAKER_MANAGER.get_speaker_names())} predefined speakers")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "device": DEVICE,
    }


@app.get("/capabilities")
async def capabilities():
    """Return backend capabilities."""
    return {
        "has_predefined_voices": True,
        "voice_cloning": True,
        "voice_cloning_formats": ["wav", "mp3", "flac"],
        "emotion_tags": False,
        "emotion_options": [],
        "streaming": False,
        "sample_rate": 22050,
        "output_format": "wav",
        "multilingual": True,
        "languages": ["en", "ky", "es"],  # Based on speaker_map
        "speaker_embedding": True,
        "embedding_dim": 128,
        "custom_params": {
            "temperature": {
                "type": "slider",
                "min": 0.1,
                "max": 1.5,
                "default": 1.0,
                "step": 0.05,
                "label": "Temperature",
            },
            "top_p": {
                "type": "slider",
                "min": 0.1,
                "max": 1.0,
                "default": 0.95,
                "step": 0.05,
                "label": "Top P",
            },
            "repetition_penalty": {
                "type": "slider",
                "min": 1.0,
                "max": 2.0,
                "default": 1.1,
                "step": 0.05,
                "label": "Repetition Penalty",
            },
            "model": {
                "type": "select",
                "options": list(MODELS.keys()) if MODELS else ["kani-2-pt-450m"],
                "default": list(MODELS.keys())[0] if MODELS else "kani-2-pt-450m",
                "label": "Model Variant",
            },
        },
    }


@app.get("/voices")
async def voices():
    """Return available voices."""
    voice_list = []
    
    # Predefined speakers from speaker_map.json
    for name in SPEAKER_MANAGER.get_speaker_names():
        # Parse language from name like "Robert (en)"
        lang = "en"
        if "(" in name and ")" in name:
            lang_code = name.split("(")[-1].replace(")", "").strip()
            lang = lang_code if lang_code else "en"
        
        voice_list.append({
            "id": f"predefined:{name}",
            "name": name,
            "gender": "UNKNOWN",
            "language": lang,
            "description": f"Predefined speaker ({lang})",
            "is_cloned": False,
        })
    
    # Model default voices (no specific speaker - random/default)
    for model_name in MODELS.keys():
        voice_list.append({
            "id": f"model:{model_name}:default",
            "name": f"{model_name} (random)",
            "gender": "UNKNOWN",
            "language": "en",
            "description": "Model default voice (random speaker)",
            "is_cloned": False,
        })
    
    return {"voices": voice_list}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Parse custom params
    custom = request.custom_params or {}
    temperature = custom.get("temperature", 1.0)
    top_p = custom.get("top_p", 0.95)
    repetition_penalty = custom.get("repetition_penalty", 1.1)
    model_name = custom.get("model", list(MODELS.keys())[0])
    
    if model_name not in MODELS:
        model_name = list(MODELS.keys())[0]
    
    model = MODELS[model_name]
    
    # Determine speaker embedding
    speaker_emb = None
    
    if request.speaker_embedding:
        # Direct embedding from request (for cloned voices)
        speaker_emb = torch.tensor([request.speaker_embedding], dtype=torch.float32)
        if DEVICE == "cuda":
            speaker_emb = speaker_emb.cuda()
    
    elif request.voice_id.startswith("predefined:"):
        # Use predefined speaker from speaker_map
        speaker_name = request.voice_id.replace("predefined:", "")
        speaker_path = SPEAKER_MANAGER.get_speaker_emb("select", speaker_name)
        if speaker_path and os.path.exists(speaker_path):
            speaker_emb = torch.load(speaker_path)
            if DEVICE == "cuda":
                speaker_emb = speaker_emb.cuda()
    
    # For model:xxx:default, speaker_emb stays None (random/default)
    
    # Generate audio
    try:
        audio, _ = model(
            request.text,
            speaker_emb=speaker_emb,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, 22050, format="WAV")
        audio_bytes = buf.getvalue()
        
        # Calculate duration
        duration = len(audio) / 22050
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": "22050",
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
    """Clone a voice from an audio sample."""
    if SPEAKER_MANAGER is None:
        raise HTTPException(status_code=503, detail="Speaker manager not loaded")
    
    try:
        # Read audio file
        audio_bytes = await audio.read()
        audio_buf = io.BytesIO(audio_bytes)
        audio_data, sr = sf.read(audio_buf)
        
        # Generate embedding using SpeakerManager (which uses SpeakerEmbedder internally)
        embedding = SPEAKER_MANAGER.generate_embedding((sr, audio_data))
        
        # Convert to list for JSON serialization
        if torch.is_tensor(embedding):
            embedding_list = embedding.cpu().squeeze().tolist()
        else:
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        # Ensure it's a flat list
        if isinstance(embedding_list[0], list):
            embedding_list = embedding_list[0]
        
        # Generate voice ID
        safe_name = name.replace(" ", "_").replace(":", "-")
        voice_id = f"cloned:{safe_name}"
        
        return {
            "id": voice_id,
            "name": name,
            "gender": "UNKNOWN",
            "language": "en",
            "description": "Cloned voice",
            "is_cloned": True,
            "embedding": embedding_list,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
