import os
import sys
import threading
import requests
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import base64
import tempfile
import subprocess
import shutil
import re
import hashlib
import wave


# Optional (Kokoro in-process)
try:
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import onnxruntime as rt  # type: ignore
    from kokoro_onnx import Kokoro as KokoroOnnx  # type: ignore
except Exception:
    np = None
    sf = None
    rt = None
    KokoroOnnx = None

# Optional (Orpheus TTS via llama.cpp - CPU-friendly)
try:
    from orpheus_cpp import OrpheusCpp  # type: ignore
except Exception:
    OrpheusCpp = None

# Optional (Misaki G2P - better phonemization for Kokoro, used by HuggingFace demo)
_MISAKI_G2P = None
try:
    from misaki import en as misaki_en  # type: ignore
    _MISAKI_G2P = misaki_en.G2P()
except Exception:
    pass

# Optional: Docker-based TTS backends (Kani, etc.)
# These run in Docker containers and communicate via HTTP
_DOCKER_BACKENDS = {}
try:
    from docker_backends.manager import DockerBackendManager, DockerBackendConfig
    _DOCKER_BACKEND_MANAGER = DockerBackendManager()
except ImportError:
    _DOCKER_BACKEND_MANAGER = None

app = Flask(__name__)

def _load_local_env_file() -> None:
    """
    Lightweight .env loader for local development.
    Existing environment variables keep precedence over file values.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Keep startup resilient if .env is malformed.
        pass


_load_local_env_file()

# Google Cloud API Key (set in .env or process environment)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()

# Preview caching
PREVIEW_SAMPLE_TEXT = "Hello! This is a preview of how I sound. I can read your entire manuscript and turn it into a beautiful audiobook."
PREVIEW_CACHE_VERSION = "v1"  # bump to invalidate all cached previews
DEFAULT_PREVIEW_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "previews"
MIN_VALID_AUDIO_BYTES = 256  # guardrail to avoid caching/serving empty or corrupted audio

# Optional: per-project pronunciation overrides (applied before synthesis).
# Format: JSON object mapping "from" -> "to", e.g. { "beautiful": "byoo-tuh-ful" }
DEFAULT_PRONUNCIATIONS_PATH = Path(__file__).resolve().parent / "pronunciations.json"

# Global state for progress tracking
conversion_state = {
    "running": False,
    "current_file": "",
    "progress": 0,
    "total": 0,
    "completed": [],
    "errors": [],
    "cancelled": False
}

# Chirp3-HD voice characters (celestial names)
CHIRP3_VOICES = [
    ('Achernar', 'FEMALE', 'Ethereal & Light'),
    ('Achird', 'MALE', 'Deep & Warm'),
    ('Algenib', 'MALE', 'Clear & Steady'),
    ('Algieba', 'MALE', 'Rich & Resonant'),
    ('Alnilam', 'MALE', 'Strong & Bold'),
    ('Aoede', 'FEMALE', 'Musical & Flowing'),
    ('Autonoe', 'FEMALE', 'Gentle & Wise'),
    ('Callirrhoe', 'FEMALE', 'Elegant & Smooth'),
    ('Charon', 'MALE', 'Deep & Mysterious'),
    ('Despina', 'FEMALE', 'Bright & Clear'),
    ('Enceladus', 'MALE', 'Powerful & Epic'),
    ('Erinome', 'FEMALE', 'Soft & Soothing'),
    ('Fenrir', 'MALE', 'Dramatic & Intense'),
    ('Gacrux', 'FEMALE', 'Warm & Inviting'),
    ('Iapetus', 'MALE', 'Ancient & Wise'),
    ('Kore', 'FEMALE', 'Youthful & Fresh'),
    ('Laomedeia', 'FEMALE', 'Noble & Graceful'),
    ('Leda', 'FEMALE', 'Sweet & Melodic'),
    ('Orus', 'MALE', 'Commanding'),
    ('Puck', 'MALE', 'Playful & Light'),
    ('Pulcherrima', 'FEMALE', 'Beautiful & Pure'),
    ('Rasalgethi', 'MALE', 'Heroic'),
    ('Sadachbia', 'MALE', 'Calm & Thoughtful'),
    ('Sadaltager', 'MALE', 'Steady & Reliable'),
    ('Schedar', 'MALE', 'Regal & Proud'),
    ('Sulafat', 'FEMALE', 'Dreamy & Soft'),
    ('Umbriel', 'MALE', 'Shadowy & Deep'),
    ('Vindemiatrix', 'FEMALE', 'Harvest & Rich'),
    ('Zephyr', 'FEMALE', 'Breezy & Light'),
    ('Zubenelgenubi', 'MALE', 'Balanced'),
]

def build_voice_dict():
    """Build comprehensive voice dictionary"""
    voices = {}
    
    # Chirp3-HD voices (newest, highest quality)
    regions = [
        ('en-US', '🇺🇸 American'),
        ('en-GB', '🇬🇧 British'),
        ('en-AU', '🇦🇺 Australian'),
        ('en-IN', '🇮🇳 Indian'),
    ]
    
    for region_code, region_name in regions:
        for voice_name, gender, desc in CHIRP3_VOICES:
            voice_id = f"{region_code}-Chirp3-HD-{voice_name}"
            voices[voice_id] = {
                'name': voice_name,
                'code': region_code,
                'gender': gender,
                'type': 'Chirp3-HD',
                'desc': desc,
                'region': region_name
            }
    
    # Studio voices (premium, designed for long-form content like audiobooks)
    studio_voices = [
        # US English
        ('en-US-Studio-M', 'Marcus', 'MALE', 'Deep & Authoritative', '🇺🇸 American'),
        ('en-US-Studio-O', 'Oliver', 'MALE', 'Warm Narrator', '🇺🇸 American'),
        ('en-US-Studio-Q', 'Quinn', 'MALE', 'Professional', '🇺🇸 American'),
        # UK English
        ('en-GB-Studio-B', 'Bernard', 'MALE', 'Distinguished British', '🇬🇧 British'),
        ('en-GB-Studio-C', 'Charlotte', 'FEMALE', 'Elegant British', '🇬🇧 British'),
    ]
    for voice_id, name, gender, desc, region in studio_voices:
        voices[voice_id] = {
            'name': name,
            'code': voice_id.split('-')[0] + '-' + voice_id.split('-')[1],
            'gender': gender,
            'type': 'Studio',
            'desc': desc,
            'region': region
        }
    
    # Neural2 voices (high quality)
    neural2_voices = [
        ('en-US-Neural2-A', 'Aria', 'FEMALE', 'Warm & Friendly'),
        ('en-US-Neural2-C', 'Clara', 'FEMALE', 'Clear & Bright'),
        ('en-US-Neural2-D', 'Daniel', 'MALE', 'Confident & Deep'),
        ('en-US-Neural2-F', 'Felix', 'MALE', 'Calm & Soothing'),
        ('en-US-Neural2-I', 'Ivy', 'FEMALE', 'Expressive'),
        ('en-US-Neural2-J', 'James', 'MALE', 'Authoritative'),
        ('en-GB-Neural2-A', 'Alice', 'FEMALE', 'Elegant'),
        ('en-GB-Neural2-B', 'Benjamin', 'MALE', 'Distinguished'),
        ('en-GB-Neural2-D', 'David', 'MALE', 'Warm & Clear'),
        ('en-AU-Neural2-A', 'Amelia', 'FEMALE', 'Friendly'),
        ('en-AU-Neural2-B', 'Bruce', 'MALE', 'Natural'),
    ]
    
    for voice_id, name, gender, desc in neural2_voices:
        code = voice_id.split('-')[0] + '-' + voice_id.split('-')[1]
        region = '🇺🇸 American' if 'US' in code else '🇬🇧 British' if 'GB' in code else '🇦🇺 Australian'
        voices[voice_id] = {
            'name': name,
            'code': code,
            'gender': gender,
            'type': 'Neural2',
            'desc': desc,
            'region': region
        }
    
    # WaveNet voices (older but cheaper - $16/1M chars vs $16 Neural2, but often bundled cheaper)
    wavenet_voices = [
        # US English
        ('en-US-Wavenet-A', 'Wavenet Aria', 'FEMALE', 'Clear'),
        ('en-US-Wavenet-B', 'Wavenet Brian', 'MALE', 'Deep'),
        ('en-US-Wavenet-C', 'Wavenet Chloe', 'FEMALE', 'Warm'),
        ('en-US-Wavenet-D', 'Wavenet Derek', 'MALE', 'Neutral'),
        ('en-US-Wavenet-E', 'Wavenet Emma', 'FEMALE', 'Soft'),
        ('en-US-Wavenet-F', 'Wavenet Frank', 'MALE', 'Friendly'),
        ('en-US-Wavenet-G', 'Wavenet Grace', 'FEMALE', 'Bright'),
        ('en-US-Wavenet-H', 'Wavenet Henry', 'MALE', 'Warm'),
        ('en-US-Wavenet-I', 'Wavenet Iris', 'FEMALE', 'Expressive'),
        ('en-US-Wavenet-J', 'Wavenet Jack', 'MALE', 'Calm'),
        # UK English
        ('en-GB-Wavenet-A', 'Wavenet Anna', 'FEMALE', 'Elegant'),
        ('en-GB-Wavenet-B', 'Wavenet Ben', 'MALE', 'Professional'),
        ('en-GB-Wavenet-C', 'Wavenet Claire', 'FEMALE', 'Refined'),
        ('en-GB-Wavenet-D', 'Wavenet David', 'MALE', 'Distinguished'),
        ('en-GB-Wavenet-F', 'Wavenet Fiona', 'FEMALE', 'Warm'),
        # Australian English
        ('en-AU-Wavenet-A', 'Wavenet Ava', 'FEMALE', 'Friendly'),
        ('en-AU-Wavenet-B', 'Wavenet Blake', 'MALE', 'Natural'),
        ('en-AU-Wavenet-C', 'Wavenet Caitlin', 'FEMALE', 'Clear'),
        ('en-AU-Wavenet-D', 'Wavenet Dylan', 'MALE', 'Casual'),
    ]
    
    for voice_id, name, gender, desc in wavenet_voices:
        code = voice_id.split('-')[0] + '-' + voice_id.split('-')[1]
        region = '🇺🇸 American' if 'US' in code else '🇬🇧 British' if 'GB' in code else '🇦🇺 Australian'
        voices[voice_id] = {
            'name': name,
            'code': code,
            'gender': gender,
            'type': 'WaveNet',
            'desc': desc,
            'region': region
        }
    
    return voices

VOICES = build_voice_dict()

KOKORO_VOICES = {}
PIPER_VOICES = {}
ORPHEUS_VOICES = {}
DOCKER_TTS_VOICES = {}  # Voices from Docker backends (Kani, etc.)

def _b64url_encode(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")

def _b64url_decode(s: str) -> str:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")

def apply_inline_pronunciation_markup(text: str) -> str:
    """
    Lightweight inline pronunciation markup.

    Supported:
      - [label](say:spoken)  -> replaces with "spoken"
      - [label](/ipa/)       -> replaces with "label" (IPA is ignored; Kokoro/Piper don't consume raw IPA)
      - [label](ipa:/ipa/)   -> replaces with "label"
    """
    if not text:
        return text

    # [label](say:spoken)  (whitespace tolerant)
    text = re.sub(
        r"\[(?P<label>[^\]]+)\]\(\s*say:(?P<say>[^)]+?)\s*\)",
        lambda m: (m.group("say") or "").strip(),
        text,
        flags=re.IGNORECASE,
    )

    # [label](/ipa/) or [label](ipa:/ipa/) -> keep label only (strip markup)
    text = re.sub(
        r"\[(?P<label>[^\]]+)\]\(\s*(?:ipa:)?\s*/(?P<ipa>[^)]+?)/\s*\)",
        lambda m: (m.group("label") or "").strip(),
        text,
        flags=re.IGNORECASE,
    )

    return text

_PRONUNCIATIONS_CACHE: dict[str, str] | None = None
_PRONUNCIATIONS_CACHE_MTIME_NS: int | None = None

def _load_pronunciations() -> dict[str, str]:
    global _PRONUNCIATIONS_CACHE
    global _PRONUNCIATIONS_CACHE_MTIME_NS

    override = os.environ.get("PRONUNCIATIONS_PATH")
    path = Path(override).expanduser().resolve() if override else DEFAULT_PRONUNCIATIONS_PATH
    try:
        st = path.stat()
        mtime_ns = getattr(st, "st_mtime_ns", None) or int(st.st_mtime * 1_000_000_000)
    except Exception:
        _PRONUNCIATIONS_CACHE = {}
        _PRONUNCIATIONS_CACHE_MTIME_NS = None
        return {}

    if _PRONUNCIATIONS_CACHE is not None and _PRONUNCIATIONS_CACHE_MTIME_NS == mtime_ns:
        return _PRONUNCIATIONS_CACHE

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}

    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
        out[k] = v

    _PRONUNCIATIONS_CACHE = out
    _PRONUNCIATIONS_CACHE_MTIME_NS = mtime_ns
    return out

def apply_pronunciation_overrides(text: str) -> str:
    """
    Apply per-word/phrase pronunciation overrides from pronunciations.json.

    This is a simple text replacement layer (not IPA). It’s most useful for
    turning a problematic word into a "say:"-style spelling (e.g. "byoo-tuh-ful").
    """
    if not text:
        return text
    mapping = _load_pronunciations()
    if not mapping:
        return text

    # Longest-first to avoid partial replacements (e.g. "New York" before "York").
    for src in sorted(mapping.keys(), key=len, reverse=True):
        dst = mapping[src]
        # Replace on token boundaries where possible (works for simple words), but
        # also handles phrases that include spaces/punctuation.
        pattern = rf"(?i)(?<!\w){re.escape(src)}(?!\w)"
        text = re.sub(pattern, dst, text)
    return text


def _find_cli_in_venv(exe_name: str) -> str | None:
    # Prefer PATH, then check the active venv and local .venv for Windows-friendly installs.
    exe = shutil.which(exe_name)
    if exe:
        return exe

    candidates: list[Path] = []
    try:
        py_dir = Path(sys.executable).resolve().parent
        candidates.extend([py_dir / exe_name, py_dir / f"{exe_name}.exe"])
    except Exception:
        pass

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_dir = Path(venv)
        candidates.extend([
            venv_dir / "Scripts" / exe_name,
            venv_dir / "Scripts" / f"{exe_name}.exe",
        ])

    local_scripts = Path(__file__).resolve().parent / ".venv" / "Scripts"
    candidates.extend([local_scripts / exe_name, local_scripts / f"{exe_name}.exe"])

    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except Exception:
            continue
    return None


def piper_cli_path() -> str | None:
    # Piper TTS CLI (typically installed via `pip install piper-tts` or a standalone binary).
    return _find_cli_in_venv("piper")

def piper_cli_command() -> list[str] | None:
    # Prefer running the module with the current interpreter; avoids broken console-script shims.
    try:
        import importlib.util

        if importlib.util.find_spec("piper") is not None:
            return [sys.executable, "-m", "piper"]
    except Exception:
        pass

    exe = piper_cli_path()
    if exe:
        return [exe]
    return None

def piper_models_dir() -> Path:
    # Allow user to keep Piper's "official" directory structure (e.g. en/en_US/...)
    # under a single base folder and let the app discover models recursively.
    override = os.environ.get("PIPER_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "models" / "piper"

def _pick_piper_config_for_model(model_path: Path) -> Path | None:
    """Pick the best matching config for a given Piper ONNX model file."""
    # Prefer <model>.onnx.json (Piper convention), then <model>.json, then any .json in folder.
    candidates = [
        Path(str(model_path) + ".json"),
        model_path.with_suffix(".json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    try:
        json_files = sorted([p for p in model_path.parent.glob("*.json") if p.is_file()])
    except Exception:
        json_files = []
    return json_files[0] if json_files else None

def build_piper_voice_dict() -> dict:
    """
    Discover Piper voices by scanning for models recursively under:
      audiobook_generator/models/piper/

    This supports keeping Piper's "official" structure, e.g.:
      models/piper/en/en_US/lessac/.../*.onnx

    Each model is a pair:
      - *.onnx
      - matching config *.onnx.json (recommended)
    """
    # If Piper isn't installed/available, don't show Piper voices in the UI.
    # (Otherwise users can select them and then synthesis fails.)
    if not piper_cli_command():
        return {}

    base = piper_models_dir()
    if not base.exists():
        return {}

    voices: dict[str, dict] = {}
    # Find all ONNX models under base (official repos can be nested multiple levels deep)
    try:
        model_files = sorted([p for p in base.rglob("*.onnx") if p.is_file()])
    except Exception:
        model_files = []

    for model_path in model_files:
        config_path = _pick_piper_config_for_model(model_path)
        rel_model = None
        try:
            rel_model = model_path.relative_to(base).as_posix()
        except Exception:
            rel_model = model_path.name

        # Use model stem as display name (typically like en_US-lessac-high)
        public_name = model_path.stem
        # Token based on relative path so multiple nested voices don't collide
        voice_token = rel_model

        desc = "Local neural TTS (Piper)"
        speaker_id_map: dict | None = None
        num_speakers: int | None = None
        try:
            if config_path and config_path.exists():
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                # Best-effort: config schema varies across voice packs.
                lang = (
                    cfg.get("language", "")
                    or cfg.get("lang", "")
                    or ((cfg.get("espeak", {}) or {}).get("voice", ""))
                )
                quality = (
                    cfg.get("quality", "")
                    or ((cfg.get("audio", {}) or {}).get("quality", ""))
                )
                if lang and quality:
                    desc = f"Local neural TTS (Piper, {lang}, {quality})"
                elif lang:
                    desc = f"Local neural TTS (Piper, {lang})"

                # Multi-speaker packs (e.g. libritts_r) expose many voices inside one model.
                # Piper's config typically contains:
                #   - num_speakers: int
                #   - speaker_id_map: { "<speaker_name>": <int_index>, ... }
                speaker_id_map = cfg.get("speaker_id_map")
                num_speakers = cfg.get("num_speakers")
        except Exception:
            pass

        # If multi-speaker, expand into individual selectable voices.
        if isinstance(speaker_id_map, dict) and isinstance(num_speakers, int) and num_speakers > 1 and len(speaker_id_map) > 1:
            # Sort by speaker index for stable ordering; keys are often dataset speaker IDs (e.g. "3922").
            for speaker_name, speaker_index in sorted(speaker_id_map.items(), key=lambda kv: kv[1]):
                try:
                    speaker_index_int = int(speaker_index)
                except Exception:
                    continue
                speaker_name_str = str(speaker_name)
                voice_token_speaker = f"{voice_token}|spk={speaker_index_int}"
                voice_id = f"piper:{_b64url_encode(voice_token_speaker)}"
                voices[voice_id] = {
                    "name": f"{public_name} • {speaker_name_str}",
                    "code": "local",
                    "gender": "UNKNOWN",
                    "type": "Piper (Offline)",
                    "desc": f"{desc} (speaker {speaker_name_str})",
                    "region": "🐦 Piper",
                    # Internal fields:
                    "_piper_model_path": str(model_path),
                    "_piper_config_path": str(config_path) if config_path else None,
                    "_piper_speaker": speaker_index_int,
                    "_piper_speaker_name": speaker_name_str,
                }
        else:
            voice_id = f"piper:{_b64url_encode(voice_token)}"
            voices[voice_id] = {
                "name": public_name,
                "code": "local",
                "gender": "UNKNOWN",
                "type": "Piper (Offline)",
                "desc": desc,
                "region": "🐦 Piper",
                # Internal fields:
                "_piper_model_path": str(model_path),
                "_piper_config_path": str(config_path) if config_path else None,
            }
    return voices

PIPER_VOICES = build_piper_voice_dict()

def is_piper_voice(voice_id: str) -> bool:
    return isinstance(voice_id, str) and voice_id.startswith("piper:")

def kokoro_cli_path() -> str | None:
    return _find_cli_in_venv("kokoro-tts")

def kokoro_default_paths():
    base = Path(__file__).resolve().parent / "models" / "kokoro"
    # Try fp16 first (better DirectML compatibility), fall back to fp32
    fp16_path = base / "kokoro-v1.0.fp16.onnx"
    fp32_path = base / "kokoro-v1.0.onnx"
    default_model = str(fp16_path) if fp16_path.exists() else str(fp32_path)
    model_path = os.environ.get("KOKORO_MODEL_PATH", default_model)
    voices_path = os.environ.get("KOKORO_VOICES_PATH", str(base / "voices-v1.0.bin"))
    return model_path, voices_path

def kokoro_preferred_providers() -> list[str]:
    """
    Provider preference for our in-process Kokoro session.
    Avoid TensorRT by default because it frequently fails without extra libs.
    """
    env_provider = os.environ.get("ONNX_PROVIDER")
    if env_provider:
        return [env_provider, "CPUExecutionProvider"]
    # Try DirectML/CUDA if available, then CPU.
    providers = []
    if rt is not None:
        available = set(rt.get_available_providers())
        for p in ["CUDAExecutionProvider", "DmlExecutionProvider"]:
            if p in available:
                providers.append(p)
    providers.append("CPUExecutionProvider")
    return providers

_KOKORO_ENGINE_LOCK = threading.Lock()
_KOKORO_ENGINE = None  # (kokoro_instance, providers)
_KOKORO_GPU_DISABLED_REASON = None  # set when GPU provider is detected to be broken at runtime

def get_kokoro_engine(prefer_gpu: bool = True):
    """
    Create/cache a Kokoro engine using a session with preferred providers.
    Returns (kokoro, active_providers).
    """
    global _KOKORO_ENGINE
    global _KOKORO_GPU_DISABLED_REASON
    if KokoroOnnx is None or rt is None:
        raise Exception("Kokoro in-process is unavailable (missing kokoro_onnx/onnxruntime/numpy/soundfile).")

    model_path, voices_path = kokoro_default_paths()
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        raise Exception(
            "Kokoro model files missing. Put kokoro-v1.0.fp16.onnx (or kokoro-v1.0.onnx) and voices-v1.0.bin in audiobook_generator/models/kokoro/ "
            "or set KOKORO_MODEL_PATH and KOKORO_VOICES_PATH."
        )

    # If we already determined the GPU provider is broken, skip it entirely.
    if prefer_gpu and _KOKORO_GPU_DISABLED_REASON is not None:
        prefer_gpu = False

    desired = kokoro_preferred_providers() if prefer_gpu else ["CPUExecutionProvider"]

    with _KOKORO_ENGINE_LOCK:
        if _KOKORO_ENGINE is not None:
            kokoro, providers = _KOKORO_ENGINE
            # If provider preference changed (e.g. user switched env var), rebuild.
            if providers == desired:
                return kokoro, providers

        sess = rt.InferenceSession(model_path, providers=desired)
        active = sess.get_providers()
        kokoro = KokoroOnnx.from_session(sess, voices_path)
        _KOKORO_ENGINE = (kokoro, desired)
        return kokoro, active

def build_kokoro_voice_dict_via_lib():
    if KokoroOnnx is None or rt is None:
        return {}
    model_path, voices_path = kokoro_default_paths()
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        return {}
    try:
        # Voice listing shouldn't require GPU; keep it simple and stable.
        kokoro, _active = get_kokoro_engine(prefer_gpu=False)
        found = list(kokoro.get_voices())
    except Exception:
        return {}

    voices = {}
    for v in sorted(set(found)):
        public_id = f"kokoro:{_b64url_encode(v)}"
        voices[public_id] = {
            "name": v,
            "code": "local",
            "gender": "UNKNOWN",
            "type": "Kokoro (Offline)",
            "desc": "Local neural TTS (Kokoro ONNX)",
            "region": "🎛️ Kokoro",
        }
    return voices

def build_kokoro_voice_dict():
    """
    Build a dict of Kokoro offline voices (via kokoro-tts CLI).
    Requires:
      - kokoro-tts in PATH (installed in the same env)
      - model files present (or KOKORO_MODEL_PATH / KOKORO_VOICES_PATH set)
    """
    # If in-process Kokoro isn't available, don't show Kokoro voices in the UI.
    # (We synthesize via the library, not the CLI.)
    if KokoroOnnx is None or rt is None:
        return {}

    # Prefer in-process listing (no CLI, no console encoding issues)
    voices = build_kokoro_voice_dict_via_lib()
    if voices:
        return voices

    # Fallback to kokoro-tts CLI if library isn't available
    # NOTE: We generally avoid CLI synthesis due to Windows console encoding issues.
    # This fallback is kept only for voice listing edge cases.
    exe = kokoro_cli_path()
    if not exe:
        return {}

    model_path, voices_path = kokoro_default_paths()
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        return {}

    try:
        cp = subprocess.run(
            [exe, "--help-voices", "--model", model_path, "--voices", voices_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return {}

    found = []
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s+([A-Za-z0-9_]+)$", line)
        if m:
            found.append(m.group(1))
    if not found:
        tokens = re.findall(r"\b[a-z]{2}_[a-z0-9_]+\b", cp.stdout or "", flags=re.IGNORECASE)
        found = list(dict.fromkeys(tokens))

    out = {}
    for v in sorted(set(found)):
        public_id = f"kokoro:{_b64url_encode(v)}"
        out[public_id] = {
            "name": v,
            "code": "local",
            "gender": "UNKNOWN",
            "type": "Kokoro (Offline)",
            "desc": "Local neural TTS (Kokoro ONNX)",
            "region": "🎛️ Kokoro",
        }
    return out

KOKORO_VOICES = build_kokoro_voice_dict()

def is_kokoro_voice(voice_id: str) -> bool:
    return isinstance(voice_id, str) and voice_id.startswith("kokoro:")

def misaki_phonemize(text: str, lang: str = "en-us") -> str | None:
    """
    Phonemize text using Misaki G2P (same as HuggingFace Kokoro demo).
    Returns None if misaki is not available, allowing fallback to espeak.
    """
    if _MISAKI_G2P is None:
        return None
    try:
        # Misaki returns (phonemes, tokens) tuple
        phonemes, _tokens = _MISAKI_G2P(text)
        return phonemes
    except Exception:
        return None

def synthesize_kokoro_wav_bytes(text: str, voice_id: str, speaking_rate: float = 1.0, lang: str = "en-us") -> bytes:
    """Synthesize speech with Kokoro in-process (offline). Falls back to CPU if provider fails."""
    if not is_kokoro_voice(voice_id):
        raise Exception("Invalid Kokoro voice id")

    voice_name = _b64url_decode(voice_id.split("kokoro:", 1)[1])

    # Use smaller chunks for stability:
    # - Avoid the kokoro-onnx "index 510 is out of bounds" edge case (token length boundary)
    # - Reduce DirectML kernel failures on long sequences
    try:
        chunk_chars = int(os.environ.get("KOKORO_CHUNK_CHARS", "500"))
    except Exception:
        chunk_chars = 500

    base_max = max(150, chunk_chars)
    chunks = split_text_into_chunks(text, base_max) if len(text) > base_max else [text]

    def _is_length_edge_case(err: Exception) -> bool:
        msg = str(err)
        return (
            "index 510 is out of bounds" in msg
            or "Context length is 510" in msg
            or "MAX_PHONEME_LENGTH" in msg
            or "Phonemes are too long" in msg
        )

    def _is_dml_kernel_error(err: Exception) -> bool:
        msg = str(err)
        return ("DmlExecutionProvider" in msg) or ("MLOperatorAuthorImpl" in msg) or ("ConvTranspose" in msg)

    def _split_aggressively(s: str) -> list[str]:
        # Fallback splitter: split by commas/semicolons, then by words.
        s = s.strip()
        if not s:
            return []
        seps = [",", ";", "—", "-", ":"]
        for sep in seps:
            if sep in s and len(s) > 1:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                if len(parts) > 1:
                    return parts
        # Word split in half
        words = s.split()
        if len(words) <= 1:
            # Hard split by chars
            mid = max(1, len(s) // 2)
            return [s[:mid].strip(), s[mid:].strip()]
        mid = len(words) // 2
        return [" ".join(words[:mid]).strip(), " ".join(words[mid:]).strip()]

    def _get_engines():
        # Prefer GPU engine if available; CPU engine always available.
        engines = []
        try:
            k_gpu, active_gpu = get_kokoro_engine(prefer_gpu=True)
            engines.append(("gpu", k_gpu, active_gpu))
        except Exception:
            pass
        k_cpu, active_cpu = get_kokoro_engine(prefer_gpu=False)
        engines.append(("cpu", k_cpu, active_cpu))
        return engines

    engines = _get_engines()

    def _render_chunk_with_retry(chunk_text: str, depth: int = 0) -> list["np.ndarray"]:
        global _KOKORO_GPU_DISABLED_REASON
        if not chunk_text.strip():
            return []
        if depth > 8:
            raise Exception("Kokoro chunk splitting exceeded max depth (text may be pathological).")

        # Try misaki phonemization first (matches HuggingFace demo, better pronunciation)
        # Falls back to espeak (kokoro-onnx default) if misaki unavailable
        phonemes = misaki_phonemize(chunk_text, lang=lang)

        last_err = None
        for label, kokoro, _active in engines:
            # If we've disabled GPU due to runtime failures, stop trying it.
            if label == "gpu" and _KOKORO_GPU_DISABLED_REASON is not None:
                continue
            try:
                # Pass phonemes if we have them; otherwise let kokoro-onnx use espeak
                samples, _sr = kokoro.create(
                    chunk_text, 
                    voice=voice_name, 
                    speed=float(speaking_rate), 
                    lang=lang,
                    phonemes=phonemes
                )
                return [np.asarray(samples, dtype=np.float32)]
            except Exception as e:
                last_err = e
                # If DirectML fails, do NOT keep retrying it via splitting.
                # Immediately allow CPU to handle the same chunk.
                if label == "gpu" and _is_dml_kernel_error(e):
                    # Cache that this provider is broken for this run, so we stop retrying it.
                    if _KOKORO_GPU_DISABLED_REASON is None:
                        _KOKORO_GPU_DISABLED_REASON = f"DML kernel failure: {str(e)[:220]}"
                    continue

                # If we hit a length/510 boundary issue, split and retry recursively.
                if _is_length_edge_case(e):
                    subparts = _split_aggressively(chunk_text)
                    if len(subparts) <= 1:
                        continue
                    rendered: list["np.ndarray"] = []
                    for sp in subparts:
                        rendered.extend(_render_chunk_with_retry(sp, depth + 1))
                    return rendered
                continue

        # If we got here, everything failed.
        raise last_err or Exception("Kokoro failed for unknown reasons.")

    # Render all chunks with per-chunk adaptive splitting and per-chunk CPU fallback.
    parts = []
    sr = None
    for ch in chunks:
        for arr in _render_chunk_with_retry(ch, depth=0):
            if sr is None:
                # KokoroOnnx uses a constant sample rate from config; safe to set once.
                sr = 24000
            parts.append(arr)

    if not parts:
        raise Exception("No audio samples generated.")

    audio = np.concatenate(parts)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    data = buf.getvalue()
    if len(data) < MIN_VALID_AUDIO_BYTES:
        raise Exception("Generated WAV is empty/invalid.")
    return data


# =============================================================================
# Orpheus TTS (LLM-based, runs on CPU via llama.cpp)
# =============================================================================

# Orpheus voices - these are built into the model
ORPHEUS_VOICE_LIST = [
    ("tara", "FEMALE", "Natural & Conversational"),
    ("leah", "FEMALE", "Warm & Friendly"),
    ("jess", "FEMALE", "Clear & Professional"),
    ("mia", "FEMALE", "Soft & Gentle"),
    ("zoe", "FEMALE", "Bright & Energetic"),
    ("leo", "MALE", "Calm & Confident"),
    ("dan", "MALE", "Deep & Authoritative"),
    ("zac", "MALE", "Youthful & Dynamic"),
]

# Orpheus emotion tags that can be used in text
ORPHEUS_EMOTION_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]

_ORPHEUS_ENGINE_LOCK = threading.Lock()
_ORPHEUS_ENGINE = None  # Cached OrpheusCpp instance


def get_orpheus_engine():
    """
    Create/cache an Orpheus engine instance.
    The model is downloaded automatically on first use (~2GB).
    """
    global _ORPHEUS_ENGINE
    if OrpheusCpp is None:
        raise Exception("Orpheus TTS is unavailable (orpheus-cpp not installed).")

    with _ORPHEUS_ENGINE_LOCK:
        if _ORPHEUS_ENGINE is None:
            _ORPHEUS_ENGINE = OrpheusCpp()
        return _ORPHEUS_ENGINE


def build_orpheus_voice_dict() -> dict:
    """
    Build a dict of Orpheus voices.
    Only returns voices if orpheus-cpp is installed.
    """
    if OrpheusCpp is None:
        return {}

    voices = {}
    for voice_name, gender, desc in ORPHEUS_VOICE_LIST:
        voice_id = f"orpheus:{_b64url_encode(voice_name)}"
        voices[voice_id] = {
            "name": voice_name.capitalize(),
            "code": "local",
            "gender": gender,
            "type": "Orpheus (Offline)",
            "desc": f"{desc} • Supports emotion tags",
            "region": "🦜 Orpheus",
            "_orpheus_voice": voice_name,
        }
    return voices


ORPHEUS_VOICES = build_orpheus_voice_dict()


def is_orpheus_voice(voice_id: str) -> bool:
    return isinstance(voice_id, str) and voice_id.startswith("orpheus:")


def synthesize_orpheus_wav_bytes(text: str, voice_id: str, speaking_rate: float = 1.0) -> bytes:
    """Synthesize speech with Orpheus TTS (offline, CPU via llama.cpp)."""
    if not is_orpheus_voice(voice_id):
        raise Exception("Invalid Orpheus voice id")

    voice_name = _b64url_decode(voice_id.split("orpheus:", 1)[1])

    # Orpheus handles long text well, but we chunk for progress/stability
    try:
        chunk_chars = int(os.environ.get("ORPHEUS_CHUNK_CHARS", "2000"))
    except Exception:
        chunk_chars = 2000

    chunk_chars = max(500, chunk_chars)
    chunks = split_text_into_chunks(text, chunk_chars) if len(text) > chunk_chars else [text]

    orpheus = get_orpheus_engine()

    # Orpheus outputs 24kHz audio
    sample_rate = 24000
    all_samples = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        try:
            # OrpheusCpp.tts returns (sample_rate, samples) where samples is int16 numpy array
            sr, samples = orpheus.tts(chunk, options={"voice_id": voice_name})
            if samples is not None and len(samples) > 0:
                all_samples.append(samples.squeeze())
                sample_rate = sr  # Use the actual sample rate from Orpheus
        except Exception as e:
            raise Exception(f"Orpheus synthesis failed: {str(e)}")

    if not all_samples:
        raise Exception("No audio samples generated by Orpheus.")

    # Concatenate all chunks
    try:
        # Import numpy if not already available (it's a dependency of orpheus-cpp)
        import numpy as np
        audio = np.concatenate(all_samples)
    except Exception:
        # Fallback: just use the first chunk
        audio = all_samples[0]

    # Convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    data = buf.getvalue()
    if len(data) < MIN_VALID_AUDIO_BYTES:
        raise Exception("Generated WAV is empty/invalid.")
    return data


# ---------------------------------------------------------------------------
# Docker-based TTS Backends (Kani, etc.)
# ---------------------------------------------------------------------------

def is_docker_tts_voice(voice_id: str) -> bool:
    """Check if a voice ID is from a Docker-based backend."""
    return isinstance(voice_id, str) and voice_id.startswith("docker:")


def register_docker_backend(backend_id: str, port: int, name: str = None):
    """
    Register a Docker TTS backend.
    
    The backend should be running and accessible at localhost:{port} with endpoints:
      - GET /health
      - GET /capabilities  
      - GET /voices
      - POST /synthesize
      - POST /clone_voice (if voice cloning supported)
    """
    global DOCKER_TTS_VOICES
    
    if _DOCKER_BACKEND_MANAGER is None:
        print(f"[docker] Backend manager not available, cannot register {backend_id}")
        return False
    
    from docker_backends.manager import DockerBackendConfig
    
    config = DockerBackendConfig(
        backend_id=backend_id,
        name=name or backend_id.title(),
        image=f"audiobook-{backend_id}:latest",
        port=port,
        gpu=False,  # Managed externally
        health_endpoint="/health",
        startup_timeout=60,
    )
    
    backend = _DOCKER_BACKEND_MANAGER.register(config)
    _DOCKER_BACKENDS[backend_id] = backend
    
    # Load voices if backend is available
    if backend.is_available():
        _refresh_docker_backend_voices(backend_id)
        return True
    else:
        print(f"[docker] Backend {backend_id} registered but not available at port {port}")
        return False


def _refresh_docker_backend_voices(backend_id: str):
    """Refresh voices from a Docker backend."""
    global DOCKER_TTS_VOICES
    
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend or not backend.is_available():
        return
    
    try:
        voices = backend.get_voices()
        caps = backend.get_capabilities()
        
        for voice in voices:
            # Create app voice ID: docker:{backend}:{voice_id}
            app_voice_id = f"docker:{backend_id}:{voice.id}"
            
            DOCKER_TTS_VOICES[app_voice_id] = {
                "name": voice.name,
                "code": "docker",
                "gender": voice.gender,
                "type": f"{backend.name} (Docker)",
                "desc": voice.description or f"From {backend.name}",
                "region": f"🐳 {backend.name}",
                "_backend_id": backend_id,
                "_backend_voice_id": voice.id,
                "_is_cloned": voice.is_cloned,
                "_embedding": voice.embedding,
                "_capabilities": caps,
            }
        
        print(f"[docker] Loaded {len(voices)} voices from {backend_id}")
    except Exception as e:
        print(f"[docker] Failed to load voices from {backend_id}: {e}")


def synthesize_docker_tts_wav_bytes(
    text: str, 
    voice_id: str, 
    speaking_rate: float = 1.0,
    custom_params: dict = None
) -> bytes:
    """Synthesize speech using a Docker-based TTS backend."""
    if not is_docker_tts_voice(voice_id):
        raise Exception("Invalid Docker TTS voice id")
    
    # Parse voice ID: docker:{backend_id}:{backend_voice_id}
    parts = voice_id.split(":", 2)
    if len(parts) != 3:
        raise Exception(f"Invalid voice id format: {voice_id}")
    
    _, backend_id, backend_voice_id = parts
    
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        raise Exception(f"Backend {backend_id} not registered")
    
    if not backend.is_available():
        raise Exception(f"Backend {backend_id} is not running")
    
    # Get voice info for embedding (if cloned voice)
    voice_info = DOCKER_TTS_VOICES.get(voice_id, {})
    embedding = voice_info.get("_embedding")
    
    from docker_backends.interface import SynthesisRequest
    
    request = SynthesisRequest(
        text=text,
        voice_id=backend_voice_id,
        speed=speaking_rate,
        speaker_embedding=embedding,
        custom_params=custom_params or {},
    )
    
    response = backend.synthesize(request)
    
    # Ensure we return WAV bytes
    if response.format == "wav":
        return response.audio_bytes
    else:
        # Convert to WAV if needed (shouldn't happen with our backends)
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(response.audio_bytes))
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()


def docker_clone_voice(backend_id: str, audio_bytes: bytes, name: str, sample_rate: int = 16000) -> dict:
    """
    Clone a voice using a Docker backend that supports voice cloning.
    
    Returns a dict with the new voice info (including embedding).
    """
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        raise Exception(f"Backend {backend_id} not registered")
    
    caps = backend.get_capabilities()
    if not caps.voice_cloning:
        raise Exception(f"Backend {backend_id} does not support voice cloning")
    
    voice = backend.clone_voice(audio_bytes, name, sample_rate)
    
    # Register the cloned voice
    app_voice_id = f"docker:{backend_id}:{voice.id}"
    
    DOCKER_TTS_VOICES[app_voice_id] = {
        "name": voice.name,
        "code": "docker",
        "gender": voice.gender,
        "type": f"{backend.name} (Cloned)",
        "desc": "Cloned voice",
        "region": f"🐳 {backend.name}",
        "_backend_id": backend_id,
        "_backend_voice_id": voice.id,
        "_is_cloned": True,
        "_embedding": voice.embedding,
    }
    
    return {
        "id": app_voice_id,
        "name": voice.name,
        "embedding": voice.embedding,
    }


# Try to connect to any pre-registered Docker backends on startup
def _init_docker_backends():
    """Check for running Docker backends and register them."""
    # Check common ports for known backends
    backends_to_check = [
        ("kani", 7862, "Kani TTS"),
        # Future backends can be added here
        # ("f5tts", 7863, "F5-TTS"),
    ]
    
    for backend_id, port, name in backends_to_check:
        try:
            # Quick health check
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                print(f"[docker] Found {name} backend on port {port}")
                register_docker_backend(backend_id, port, name)
        except Exception:
            pass  # Backend not running, that's fine

# Initialize on module load (non-blocking, just checks)
try:
    _init_docker_backends()
except Exception as e:
    print(f"[docker] Backend initialization failed: {e}")


def synthesize_google_mp3_bytes(text: str, voice_id: str, speaking_rate: float = 1.0) -> bytes:
    """Synthesize speech using Google Cloud TTS REST API (online)."""
    if not GOOGLE_API_KEY:
        raise Exception(
            "Google API key is not configured. Set GOOGLE_API_KEY in .env or the environment."
        )

    voice_info = VOICES.get(voice_id, VOICES["en-GB-Chirp3-HD-Enceladus"])

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": voice_info["code"], "name": voice_id},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": speaking_rate, "pitch": 0.0},
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        try:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
        except Exception:
            error_msg = response.text or "Unknown error"
        raise Exception(f"API Error: {error_msg}")

    return base64.b64decode(response.json()["audioContent"])


def synthesize_piper_wav_bytes(text: str, voice_id: str, speaking_rate: float = 1.0) -> bytes:
    """Synthesize speech using Piper (offline) via the `piper` CLI, returning WAV bytes."""
    if not is_piper_voice(voice_id):
        raise Exception("Invalid Piper voice id")
    cmd_base = piper_cli_command()
    if not cmd_base:
        raise Exception("Piper CLI not found. Install with: python -m pip install piper-tts (or add piper to PATH).")

    voice_info = PIPER_VOICES.get(voice_id)
    if not voice_info:
        raise Exception("Piper voice not found (models missing?).")

    model_path = voice_info.get("_piper_model_path")
    config_path = voice_info.get("_piper_config_path")
    speaker_id = voice_info.get("_piper_speaker")
    if not model_path or not os.path.exists(model_path):
        raise Exception("Piper model file missing.")
    if config_path and (not os.path.exists(config_path)):
        config_path = None

    # Piper can handle fairly long text, but chunking improves reliability and keeps latency predictable.
    try:
        chunk_chars = int(os.environ.get("PIPER_CHUNK_CHARS", "1500"))
    except Exception:
        chunk_chars = 1500
    chunk_chars = max(300, chunk_chars)
    chunks = split_text_into_chunks(text, chunk_chars) if len(text) > chunk_chars else [text]

    # UI speaking_rate is a multiplier (>1 = faster). Piper uses length_scale (>1 = slower).
    # A simple inversion works well in practice.
    try:
        length_scale = max(0.5, min(2.0, 1.0 / float(speaking_rate)))
    except Exception:
        length_scale = 1.0

    out_buf = io.BytesIO()
    writer = None
    expected_params = None

    def _format_piper_error(cmd: list[str], err: subprocess.CalledProcessError) -> str:
        stderr = (err.stderr or "").strip()
        stdout = (err.stdout or "").strip()
        msg = "\n".join([m for m in [stderr, stdout] if m])
        if not msg:
            msg = "No output captured from Piper."
        return f"Piper failed (exit code {err.returncode}). {msg}\nCommand: {' '.join(cmd)}"

    def _summarize_output(label: str, text_value: str) -> str:
        if not text_value:
            return ""
        text_value = text_value.strip()
        if len(text_value) > 800:
            text_value = text_value[:800] + "..."
        return f"{label}: {text_value}"

    for ch in chunks:
        if not ch.strip():
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        try:
            cmd = [*cmd_base, "--model", model_path, "--output_file", tmp_path]
            if config_path:
                cmd.extend(["--config", config_path])
            if speaker_id is not None:
                cmd.extend(["--speaker", str(speaker_id)])
            # Optional: speed control (best-effort; older pipers may not support it)
            cmd.extend(["--length_scale", f"{length_scale:.3f}"])

            try:
                cp = subprocess.run(
                    cmd,
                    input=ch,
                    text=True,
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                # Retry without --length_scale if this piper build doesn't support it
                msg = (e.stderr or "") + "\n" + (e.stdout or "")
                if ("unrecognized arguments" in msg) and ("--length_scale" in msg):
                    cmd = [*cmd_base, "--model", model_path, "--output_file", tmp_path]
                    if config_path:
                        cmd.extend(["--config", config_path])
                    if speaker_id is not None:
                        cmd.extend(["--speaker", str(speaker_id)])
                    cp = subprocess.run(
                        cmd,
                        input=ch,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                else:
                    raise Exception(_format_piper_error(cmd, e))

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                notes = [
                    _summarize_output("stderr", cp.stderr or ""),
                    _summarize_output("stdout", cp.stdout or ""),
                ]
                note_text = "\n".join([n for n in notes if n])
                if note_text:
                    note_text = "\n" + note_text
                raise Exception(
                    f"Piper produced no output file (exit code {cp.returncode}).{note_text}"
                )

            # Append WAV frames into a single WAV stream
            with wave.open(tmp_path, "rb") as r:
                params = r.getparams()
                frames = r.readframes(r.getnframes())
            if expected_params is None:
                expected_params = params
                writer = wave.open(out_buf, "wb")
                writer.setnchannels(params.nchannels)
                writer.setsampwidth(params.sampwidth)
                writer.setframerate(params.framerate)
            else:
                # Basic consistency check
                if (params.nchannels, params.sampwidth, params.framerate) != (
                    expected_params.nchannels,
                    expected_params.sampwidth,
                    expected_params.framerate,
                ):
                    raise Exception("Piper produced inconsistent WAV parameters across chunks.")
            writer.writeframes(frames)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if writer is not None:
        writer.close()

    data = out_buf.getvalue()
    if len(data) < MIN_VALID_AUDIO_BYTES:
        raise Exception("Piper generated an empty/invalid WAV.")
    return data

def synthesize_speech(text: str, voice_id: str, speaking_rate: float = 1.0):
    """
    Returns: (audio_bytes, mimetype, extension)
    """
    text = apply_pronunciation_overrides(text)
    text = apply_inline_pronunciation_markup(text)
    if is_kokoro_voice(voice_id):
        return synthesize_kokoro_wav_bytes(text, voice_id, speaking_rate), "audio/wav", ".wav"
    if is_piper_voice(voice_id):
        return synthesize_piper_wav_bytes(text, voice_id, speaking_rate), "audio/wav", ".wav"
    if is_orpheus_voice(voice_id):
        return synthesize_orpheus_wav_bytes(text, voice_id, speaking_rate), "audio/wav", ".wav"
    if is_docker_tts_voice(voice_id):
        return synthesize_docker_tts_wav_bytes(text, voice_id, speaking_rate), "audio/wav", ".wav"
    return synthesize_google_mp3_bytes(text, voice_id, speaking_rate), "audio/mpeg", ".mp3"

def get_preview_cache_dir() -> Path:
    # Allow override for where cache is stored
    override = os.environ.get("PREVIEW_CACHE_DIR")
    base = Path(override).expanduser().resolve() if override else DEFAULT_PREVIEW_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base

def preview_cache_key(voice_id: str, speaking_rate: float) -> str:
    # Include a version + sample text so changes invalidate cache automatically.
    payload = f"{PREVIEW_CACHE_VERSION}|{voice_id}|{float(speaking_rate):.3f}|{PREVIEW_SAMPLE_TEXT}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
    # Atomic on Windows when within same filesystem
    os.replace(str(tmp_path), str(target_path))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/voices')
def list_voices():
    """Get list of available voices"""
    voices = []
    for voice_id, voice_info in VOICES.items():
        voices.append({
            'id': voice_id,
            'name': voice_info['name'],
            'gender': voice_info['gender'],
            'type': voice_info['type'],
            'desc': voice_info['desc'],
            'code': voice_info['code'],
            'region': voice_info.get('region', '')
        })

    
    # Add Kokoro voices (offline neural) if available
    for voice_id, voice_info in KOKORO_VOICES.items():
        voices.append({
            'id': voice_id,
            'name': voice_info['name'],
            'gender': voice_info.get('gender', 'UNKNOWN'),
            'type': voice_info['type'],
            'desc': voice_info['desc'],
            'code': voice_info.get('code', 'local'),
            'region': voice_info.get('region', '🎛️ Kokoro')
        })

    # Add Piper voices (offline neural) if available
    for voice_id, voice_info in PIPER_VOICES.items():
        voices.append({
            'id': voice_id,
            'name': voice_info['name'],
            'gender': voice_info.get('gender', 'UNKNOWN'),
            'type': voice_info['type'],
            'desc': voice_info['desc'],
            'code': voice_info.get('code', 'local'),
            'region': voice_info.get('region', '🐦 Piper')
        })

    # Add Orpheus voices (LLM-based offline) if available
    for voice_id, voice_info in ORPHEUS_VOICES.items():
        voices.append({
            'id': voice_id,
            'name': voice_info['name'],
            'gender': voice_info.get('gender', 'UNKNOWN'),
            'type': voice_info['type'],
            'desc': voice_info['desc'],
            'code': voice_info.get('code', 'local'),
            'region': voice_info.get('region', '🦜 Orpheus')
        })

    # Add Docker TTS voices (Kani, etc.) if available
    for voice_id, voice_info in DOCKER_TTS_VOICES.items():
        voices.append({
            'id': voice_id,
            'name': voice_info['name'],
            'gender': voice_info.get('gender', 'UNKNOWN'),
            'type': voice_info['type'],
            'desc': voice_info['desc'],
            'code': voice_info.get('code', 'docker'),
            'region': voice_info.get('region', '🐳 Docker')
        })

    # Group voices
    grouped = {
        'chirp3_us': [v for v in voices if v['type'] == 'Chirp3-HD' and v['code'] == 'en-US'],
        'chirp3_uk': [v for v in voices if v['type'] == 'Chirp3-HD' and v['code'] == 'en-GB'],
        'chirp3_au': [v for v in voices if v['type'] == 'Chirp3-HD' and v['code'] == 'en-AU'],
        'chirp3_in': [v for v in voices if v['type'] == 'Chirp3-HD' and v['code'] == 'en-IN'],
        'studio': [v for v in voices if v['type'] == 'Studio'],
        'neural2': [v for v in voices if v['type'] == 'Neural2'],
        'wavenet': [v for v in voices if v['type'] == 'WaveNet'],
        'kokoro_offline': [v for v in voices if v['type'] == 'Kokoro (Offline)'],
        'piper_offline': [v for v in voices if v['type'] == 'Piper (Offline)'],
        'orpheus_offline': [v for v in voices if v['type'] == 'Orpheus (Offline)'],
        'docker_tts': [v for v in voices if v['code'] == 'docker'],
    }
    
    return jsonify(grouped)

@app.route('/api/preview/<path:voice_id>')
def preview_voice(voice_id):
    """Generate a voice preview"""
    if voice_id not in VOICES and voice_id not in KOKORO_VOICES and voice_id not in PIPER_VOICES and voice_id not in ORPHEUS_VOICES and voice_id not in DOCKER_TTS_VOICES:
        return jsonify({'error': 'Voice not found'}), 404
    
    # Preview uses the same sample text every time; cache on disk for speed/cost.
    speaking_rate = 1.0
    cache_dir = get_preview_cache_dir()
    key = preview_cache_key(voice_id, speaking_rate)
    
    try:
        # If cached, serve directly
        # Note: extension depends on backend (mp3 for Google, wav for offline).
        # We probe for either.
        cached_mp3 = cache_dir / f"{key}.mp3"
        cached_wav = cache_dir / f"{key}.wav"
        if cached_mp3.exists() and cached_mp3.stat().st_size >= MIN_VALID_AUDIO_BYTES:
            return send_file(str(cached_mp3), mimetype="audio/mpeg", as_attachment=False)
        if cached_wav.exists() and cached_wav.stat().st_size >= MIN_VALID_AUDIO_BYTES:
            return send_file(str(cached_wav), mimetype="audio/wav", as_attachment=False)
        # If we have a cached zero-byte (or tiny) file from a prior failure, delete it.
        for bad in (cached_mp3, cached_wav):
            try:
                if bad.exists() and bad.stat().st_size < MIN_VALID_AUDIO_BYTES:
                    bad.unlink()
            except Exception:
                pass

        audio_content, mimetype, ext = synthesize_speech(PREVIEW_SAMPLE_TEXT, voice_id, speaking_rate)
        if len(audio_content) < MIN_VALID_AUDIO_BYTES:
            raise Exception("Preview generation produced empty/invalid audio (refusing to cache).")
        target = cache_dir / f"{key}{ext}"
        # Best-effort write; if it fails, still return the audio.
        try:
            if not target.exists():
                atomic_write_bytes(target, audio_content)
        except Exception:
            pass
        return send_file(
            io.BytesIO(audio_content),
            mimetype=mimetype,
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diagnostics')
def diagnostics():
    """Lightweight runtime diagnostics for troubleshooting (no secrets)."""
    info = {
        "python_version": getattr(sys, "version", ""),
        "python_executable": getattr(sys, "executable", ""),
        "onnx_provider_env": os.environ.get("ONNX_PROVIDER"),
        "kokoro_cli_found": bool(kokoro_cli_path()),
        "piper_cli_found": bool(piper_cli_command()),
        "piper_cli_path": piper_cli_path(),
        "piper_cli_command": piper_cli_command(),
        "piper_models_dir": str(piper_models_dir()),
        "piper_voices_found": len(PIPER_VOICES),
        "kokoro_model_path": kokoro_default_paths()[0],
        "kokoro_voices_path": kokoro_default_paths()[1],
        "kokoro_model_exists": os.path.exists(kokoro_default_paths()[0]),
        "kokoro_voices_exists": os.path.exists(kokoro_default_paths()[1]),
        "kokoro_gpu_disabled_reason": _KOKORO_GPU_DISABLED_REASON,
    }
    try:
        import importlib.metadata as im
        info["piper_tts_version"] = im.version("piper-tts")
    except Exception as e:
        info["piper_tts_version_error"] = str(e)
    try:
        import piper  # type: ignore
        info["piper_module_file"] = getattr(piper, "__file__", None)
    except Exception as e:
        info["piper_module_error"] = str(e)
    try:
        import onnxruntime as rt  # type: ignore
        info["onnxruntime_version"] = getattr(rt, "__version__", None)
        info["onnxruntime_file"] = getattr(rt, "__file__", None)
        info["onnxruntime_available_providers"] = rt.get_available_providers()
        try:
            info["onnxruntime_device"] = rt.get_device()
        except Exception:
            pass

        # If we have the Kokoro model locally, try creating a session and report
        # which providers are actually used. This catches cases where CUDA is
        # "available" but can't initialize and ORT falls back.
        model_path, _voices_path = kokoro_default_paths()
        if os.path.exists(model_path):
            env_provider = os.environ.get("ONNX_PROVIDER")
            if env_provider:
                # If user explicitly set a provider, DO NOT add other providers.
                # (Adding TensorRT can cause noisy failures + CPU fallback.)
                desired = [env_provider, "CPUExecutionProvider"]
            else:
                # Prefer CUDA, then DirectML, then CPU. Avoid TensorRT unless
                # the user explicitly requests it (it often requires extra libs).
                desired = []
                for p in ["CUDAExecutionProvider", "DmlExecutionProvider"]:
                    if p in info.get("onnxruntime_available_providers", []):
                        desired.append(p)
                desired.append("CPUExecutionProvider")
            try:
                sess = rt.InferenceSession(model_path, providers=desired)
                info["kokoro_session_requested_providers"] = desired
                info["kokoro_session_active_providers"] = sess.get_providers()
            except Exception as e:
                info["kokoro_session_requested_providers"] = desired
                info["kokoro_session_error"] = str(e)
    except Exception as e:
        info["onnxruntime_error"] = str(e)

    # Orpheus TTS diagnostics
    info["orpheus_available"] = OrpheusCpp is not None
    info["orpheus_voices_found"] = len(ORPHEUS_VOICES)
    if OrpheusCpp is not None:
        try:
            import importlib.metadata as im
            info["orpheus_cpp_version"] = im.version("orpheus-cpp")
        except Exception as e:
            info["orpheus_cpp_version_error"] = str(e)
        try:
            import llama_cpp
            info["llama_cpp_python_version"] = getattr(llama_cpp, "__version__", "unknown")
        except Exception as e:
            info["llama_cpp_python_error"] = str(e)

    return jsonify(info)

@app.route('/api/browse', methods=['POST'])
def browse_folder():
    """List text files in a folder"""
    data = request.json
    folder_path = data.get('path', '')
    
    if not folder_path:
        if os.name == 'nt':
            import string
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return jsonify({'type': 'drives', 'items': drives})
        else:
            folder_path = os.path.expanduser('~')
    
    folder_path = os.path.abspath(folder_path)
    
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Path does not exist'}), 400
    
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path)
    
    items = []
    try:
        for item in sorted(os.listdir(folder_path)):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                items.append({'name': item, 'type': 'folder', 'path': item_path})
            elif item.lower().endswith('.txt'):
                items.append({'name': item, 'type': 'file', 'path': item_path})
    except PermissionError:
        return jsonify({'error': 'Permission denied'}), 403
    
    parent = os.path.dirname(folder_path) if folder_path != os.path.dirname(folder_path) else None
    
    return jsonify({
        'type': 'listing',
        'current': folder_path,
        'parent': parent,
        'items': items
    })

@app.route('/api/convert', methods=['POST'])
def start_conversion():
    """Start converting text files to audio"""
    global conversion_state
    
    if conversion_state['running']:
        return jsonify({'error': 'Conversion already in progress'}), 400
    
    data = request.json
    files = data.get('files', [])
    voice_id = data.get('voice', 'en-GB-Chirp3-HD-Enceladus')
    output_folder = data.get('output_folder', '')
    speaking_rate = data.get('rate', 1.0)
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    if not output_folder:
        output_folder = os.path.dirname(files[0])
    
    output_folder = os.path.join(output_folder, 'audiobook_output')
    os.makedirs(output_folder, exist_ok=True)
    
    conversion_state = {
        "running": True,
        "current_file": "",
        "progress": 0,
        "total": len(files),
        "completed": [],
        "errors": [],
        "cancelled": False,
        "output_folder": output_folder
    }
    
    thread = threading.Thread(target=convert_files_thread, args=(files, voice_id, output_folder, speaking_rate))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'output_folder': output_folder})

def convert_single_file(input_path, output_path, voice_id, speaking_rate):
    """Convert a single text file to audio with parallel chunk processing"""
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = text.strip()
    if not text:
        raise ValueError("File is empty")
    
    # Offline voices: generate as a single WAV (no API limits)
    if is_kokoro_voice(voice_id) or is_piper_voice(voice_id) or is_orpheus_voice(voice_id):
        audio_content, _mimetype, _ext = synthesize_speech(text, voice_id, speaking_rate)
        with open(output_path, "wb") as f:
            f.write(audio_content)
        return

    # Google voices: chunk to avoid API limits, join MP3 bytes
    max_chars = 4800  # Slightly increased
    if len(text) <= max_chars:
        audio_content, _mimetype, _ext = synthesize_speech(text, voice_id, speaking_rate)
    else:
        chunks = split_text_into_chunks(text, max_chars)
        num_chunks = len(chunks)
        audio_parts = [None] * num_chunks  # Pre-allocate to maintain order

        def process_chunk(index, chunk_text):
            if conversion_state["cancelled"]:
                return index, None
            chunk_bytes, _m, _e = synthesize_speech(chunk_text, voice_id, speaking_rate)
            return index, chunk_bytes

        # Process chunks in parallel (default up to 5 concurrent requests)
        # Override with GOOGLE_CHUNK_WORKERS to tune throughput / avoid rate limits.
        try:
            chunk_worker_cap = int(os.environ.get("GOOGLE_CHUNK_WORKERS", "5"))
        except Exception:
            chunk_worker_cap = 5
        max_workers = min(max(1, chunk_worker_cap), num_chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                if conversion_state["cancelled"]:
                    raise Exception("Cancelled")
                index, audio_data = future.result()
                if audio_data:
                    audio_parts[index] = audio_data

        if None in audio_parts:
            raise Exception("Some chunks failed to generate")

        audio_content = b"".join(audio_parts)

    with open(output_path, "wb") as f:
        f.write(audio_content)

def split_text_into_chunks(text, max_chars):
    """Split text into chunks at sentence boundaries"""
    chunks = []
    current_chunk = ""
    
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?' and len(current_sentence) > 1:
            sentences.append(current_sentence)
            current_sentence = ""
    
    if current_sentence:
        sentences.append(current_sentence)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_single_file_wrapper(args):
    """Wrapper for parallel file processing"""
    file_path, output_folder, voice_id, speaking_rate = args
    filename = os.path.basename(file_path)
    output_ext = ".wav" if (is_kokoro_voice(voice_id) or is_piper_voice(voice_id) or is_orpheus_voice(voice_id) or is_docker_tts_voice(voice_id)) else ".mp3"
    output_name = os.path.splitext(filename)[0] + output_ext
    output_path = os.path.join(output_folder, output_name)
    
    try:
        convert_single_file(file_path, output_path, voice_id, speaking_rate)
        return {'success': True, 'input': filename, 'output': output_name}
    except Exception as e:
        return {'success': False, 'file': filename, 'error': str(e)}

def convert_files_thread(files, voice_id, output_folder, speaking_rate):
    """Thread function to convert files with parallel processing"""
    global conversion_state
    
    try:
        # File-level parallelism:
        # - For Google voices: default 2 (avoid API throttling)
        # - For offline voices (Kokoro/Piper/Orpheus/Docker): default 1 (avoid CPU oversubscription / GPU contention)
        is_offline = is_kokoro_voice(voice_id) or is_piper_voice(voice_id) or is_orpheus_voice(voice_id) or is_docker_tts_voice(voice_id)
        default_workers = "1" if is_offline else "2"
        env_key = "OFFLINE_FILE_WORKERS" if is_offline else "GOOGLE_FILE_WORKERS"
        try:
            file_worker_cap = int(os.environ.get(env_key, default_workers))
        except Exception:
            file_worker_cap = int(default_workers)
        max_file_workers = min(max(1, file_worker_cap), len(files))
        completed_count = 0
        
        args_list = [(f, output_folder, voice_id, speaking_rate) for f in files]
        
        with ThreadPoolExecutor(max_workers=max_file_workers) as executor:
            futures = {executor.submit(process_single_file_wrapper, args): args[0] for args in args_list}
            
            for future in as_completed(futures):
                if conversion_state['cancelled']:
                    break
                
                result = future.result()
                completed_count += 1
                conversion_state['progress'] = completed_count
                conversion_state['current_file'] = f"Processing... ({completed_count}/{len(files)})"
                
                if result['success']:
                    conversion_state['completed'].append({
                        'input': result['input'],
                        'output': result['output']
                    })
                else:
                    conversion_state['errors'].append({
                        'file': result['file'],
                        'error': result['error']
                    })
        
        conversion_state['progress'] = len(files)
    finally:
        conversion_state['running'] = False
        conversion_state['current_file'] = ""

@app.route('/api/status')
def get_status():
    """Get current conversion status"""
    return jsonify(conversion_state)

@app.route('/api/cancel', methods=['POST'])
def cancel_conversion():
    """Cancel ongoing conversion"""
    global conversion_state
    conversion_state['cancelled'] = True
    return jsonify({'status': 'cancelling'})


# ---------------------------------------------------------------------------
# Docker Backend API endpoints
# ---------------------------------------------------------------------------

@app.route('/api/docker/backends')
def list_docker_backends():
    """List registered Docker TTS backends and their status."""
    backends = []
    for backend_id, backend in _DOCKER_BACKENDS.items():
        backends.append({
            'id': backend_id,
            'name': backend.name,
            'port': backend.config.port,
            'available': backend.is_available(),
            'voice_count': len([v for v in DOCKER_TTS_VOICES if v.startswith(f"docker:{backend_id}:")]),
        })
    return jsonify({'backends': backends})


@app.route('/api/docker/backends/<backend_id>/capabilities')
def get_docker_backend_capabilities(backend_id):
    """Get capabilities of a Docker backend."""
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        return jsonify({'error': f'Backend {backend_id} not found'}), 404
    
    if not backend.is_available():
        return jsonify({'error': f'Backend {backend_id} is not running'}), 503
    
    caps = backend.get_capabilities()
    return jsonify({
        'has_predefined_voices': caps.has_predefined_voices,
        'voice_cloning': caps.voice_cloning,
        'voice_cloning_formats': caps.voice_cloning_formats,
        'emotion_tags': caps.emotion_tags,
        'emotion_options': caps.emotion_options,
        'streaming': caps.streaming,
        'sample_rate': caps.sample_rate,
        'multilingual': caps.multilingual,
        'languages': caps.languages,
        'speaker_embedding': caps.speaker_embedding,
        'embedding_dim': caps.embedding_dim,
        'custom_params': caps.custom_params,
    })


@app.route('/api/docker/backends/<backend_id>/refresh', methods=['POST'])
def refresh_docker_backend_voices(backend_id):
    """Refresh voices from a Docker backend."""
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        return jsonify({'error': f'Backend {backend_id} not found'}), 404
    
    if not backend.is_available():
        return jsonify({'error': f'Backend {backend_id} is not running'}), 503
    
    _refresh_docker_backend_voices(backend_id)
    voice_count = len([v for v in DOCKER_TTS_VOICES if v.startswith(f"docker:{backend_id}:")])
    
    return jsonify({'status': 'ok', 'voice_count': voice_count})


@app.route('/api/docker/clone', methods=['POST'])
def clone_voice_endpoint():
    """
    Clone a voice using a Docker backend.
    
    Expects multipart form data:
      - backend_id: ID of the Docker backend to use
      - name: Name for the cloned voice
      - audio: Audio file to clone from (WAV, MP3, etc.)
    """
    backend_id = request.form.get('backend_id')
    name = request.form.get('name')
    audio_file = request.files.get('audio')
    
    if not backend_id or not name or not audio_file:
        return jsonify({'error': 'Missing required fields: backend_id, name, audio'}), 400
    
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        return jsonify({'error': f'Backend {backend_id} not found'}), 404
    
    if not backend.is_available():
        return jsonify({'error': f'Backend {backend_id} is not running'}), 503
    
    caps = backend.get_capabilities()
    if not caps.voice_cloning:
        return jsonify({'error': f'Backend {backend_id} does not support voice cloning'}), 400
    
    try:
        audio_bytes = audio_file.read()
        result = docker_clone_voice(backend_id, audio_bytes, name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/docker/register', methods=['POST'])
def register_docker_backend_endpoint():
    """
    Manually register a Docker backend.
    
    JSON body:
      - backend_id: Unique ID for the backend
      - port: Port the backend is running on
      - name: Display name (optional)
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400
    
    backend_id = data.get('backend_id')
    port = data.get('port')
    name = data.get('name')
    
    if not backend_id or not port:
        return jsonify({'error': 'Missing required fields: backend_id, port'}), 400
    
    try:
        success = register_docker_backend(backend_id, int(port), name)
        if success:
            voice_count = len([v for v in DOCKER_TTS_VOICES if v.startswith(f"docker:{backend_id}:")])
            return jsonify({'status': 'ok', 'voice_count': voice_count})
        else:
            return jsonify({'error': f'Backend registered but not available on port {port}'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  AUDIOBOOK GENERATOR")
    print("  Powered by Google Cloud TTS")
    print(f"  {len(VOICES)} cloud voices (incl. Chirp3-HD)")
    print(f"  {len(KOKORO_VOICES)} Kokoro voices, {len(PIPER_VOICES)} Piper voices")
    print(f"  {len(ORPHEUS_VOICES)} Orpheus voices, {len(DOCKER_TTS_VOICES)} Docker TTS voices")
    if _DOCKER_BACKENDS:
        print(f"  Docker backends: {', '.join(_DOCKER_BACKENDS.keys())}")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
