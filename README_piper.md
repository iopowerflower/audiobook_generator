# Piper (Offline Neural TTS) Setup

Piper is a high-quality **offline** TTS engine that can run locally (CPU, and sometimes GPU depending on how you run ONNX Runtime).

This app integrates Piper via the **`piper` CLI** and auto-discovers voices by scanning for `*.onnx` models **recursively** under `audiobook_generator/models/piper/`.

## Where do Piper voices come from?

Piper **does not ship with voices**. You download model files separately.

The canonical voice list + download links live here:

- `https://github.com/rhasspy/piper/blob/master/VOICES.md`

There are also browsable docs:

- `https://tderflinger.github.io/piper-docs/about/voices/`

What you download is typically either:

- a **`.tar.gz`** per voice (containing `*.onnx` and `*.onnx.json`), or
- the two files directly: **`<voice>.onnx`** + **`<voice>.onnx.json`**

> Licensing varies by voice. Check the voice’s model card / docs before using it commercially.

## 1) Install Piper (in the same venv as the app)

From `audiobook_generator/`:

```bash
. .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Confirm the CLI is available:

```bash
piper --help
```

## 2) Add Piper model files

You can keep the **official Piper structure** exactly as-is under `models/piper/`, for example the Hugging Face tree:

- [`rhasspy/piper-voices` → `en/en_US`](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US)

Example layout:

```
audiobook_generator/
  models/
    piper/
      en/
        en_US/
          lessac/
            ...
              *.onnx
              *.onnx.json
```

Rules:
- Each voice model is a `*.onnx` plus a matching `*.onnx.json` in the same folder.
- The app will discover models even if they are nested multiple directories deep.

### If your download is a `.tar.gz`

Extract it, then place the resulting `*.onnx` + `*.onnx.json` into a single folder under:

- `audiobook_generator/models/piper/<some_voice_name>/`

Restart the app after adding models (voice list is built on startup).

## 3) Verify it shows up

- Open the UI and look for **🐦 Piper (Offline Neural)**
- Or hit: `http://127.0.0.1:5000/api/diagnostics` and check `piper_voices_found`

## Notes / tuning

- Chunk size: set `PIPER_CHUNK_CHARS` (default: 1500)
- Speed: UI “Speed” attempts to map to Piper `--length_scale` (best-effort; older pipers may not support it)


