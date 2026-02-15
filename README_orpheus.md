# Orpheus TTS (LLM-based Offline TTS) Setup


COMPLETELY UNTESTED, UNFINISHED, AND PROBABLY BROKEN!


Orpheus TTS is a state-of-the-art text-to-speech system built on a Llama-3B backbone. It produces highly natural, human-like speech with emotion support.

## Features

- **Human-like speech**: Natural intonation, emotion, and rhythm
- **8 built-in voices**: Tara, Leah, Jess, Mia, Zoe (female) + Leo, Dan, Zac (male)
- **Emotion tags**: Add `<laugh>`, `<sigh>`, `<chuckle>`, `<gasp>`, etc. to your text
- **Runs on CPU**: Uses llama.cpp for efficient CPU inference (no GPU required)
- **~2GB model**: Downloaded automatically on first use

## Requirements

- **Python 3.10, 3.11, or 3.12** (not 3.13+)
- ~2GB disk space for the model (auto-downloaded)
- ~4GB RAM recommended

## Installation

### Option 1: Use the Setup Script (Recommended)

The setup scripts automatically install Orpheus and all dependencies:

```bash
# Standard (CPU) setup:
bash run_app.sh

# Or with CUDA/GPU support:
bash run_app_cuda.sh
```

The script will:
- Create/activate the virtual environment
- Install all Python dependencies
- Install `orpheus-cpp` and `llama-cpp-python` with pre-built wheels (no compiler needed)

### Option 2: Manual Installation

If you prefer to install manually:

**Step 1: Install orpheus-cpp**
```bash
pip install orpheus-cpp
```

**Step 2: Install llama-cpp-python**

The `llama-cpp-python` package must be installed separately with pre-built wheels:

**Windows / Linux (CPU):**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**macOS with Apple Silicon (Metal acceleration):**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

**Windows / Linux with NVIDIA GPU (CUDA):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### Verify Installation

Start the app and check the diagnostics endpoint:

```bash
python app.py
# Open http://localhost:5000/api/diagnostics
```

Look for:
- `orpheus_available: true`
- `orpheus_voices_found: 8`

## Usage

Once installed, Orpheus voices will appear in the **"Orpheus (LLM TTS)"** tab in the voice selector.

### Available Voices

| Voice | Gender | Style |
|-------|--------|-------|
| Tara | Female | Natural & Conversational |
| Leah | Female | Warm & Friendly |
| Jess | Female | Clear & Professional |
| Mia | Female | Soft & Gentle |
| Zoe | Female | Bright & Energetic |
| Leo | Male | Calm & Confident |
| Dan | Male | Deep & Authoritative |
| Zac | Male | Youthful & Dynamic |

### Emotion Tags

You can add emotion tags directly in your text for more expressive speech:

```
I can't believe it worked! <laugh> This is amazing.
<sigh> I really thought we had more time.
Wait, what was that? <gasp>
```

**Supported tags:**
- `<laugh>` - Laughter
- `<chuckle>` - Soft laughter
- `<sigh>` - Sighing
- `<cough>` - Coughing
- `<sniffle>` - Sniffling
- `<groan>` - Groaning
- `<yawn>` - Yawning
- `<gasp>` - Gasping

## First Run - Model Download

On first use, Orpheus will automatically download the model (~2GB). This may take a few minutes depending on your connection. The model is cached for future use.

You'll see progress in the terminal:
```
Downloading orpheus model...
```

## Performance Notes

- **First generation is slow**: The model needs to load into memory (~30-60 seconds)
- **Subsequent generations are faster**: ~2-5x real-time on modern CPUs
- **Long texts are chunked**: For stability, long texts are split into ~2000 character chunks

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORPHEUS_CHUNK_CHARS` | `2000` | Max characters per synthesis chunk |

## Troubleshooting

### "orpheus-cpp not installed"

Make sure you installed both packages:
```bash
pip install orpheus-cpp
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### "llama_cpp not found" or import errors

The `llama-cpp-python` package needs to be compiled for your platform. Try:
```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall
```

### Very slow generation

- Orpheus is an LLM-based TTS, so it's naturally slower than ONNX models like Kokoro
- First run loads the model, subsequent runs are faster
- Consider using shorter text chunks
- For faster inference, use Kokoro or Piper for long audiobooks

### Out of memory

- Orpheus needs ~4GB RAM to run comfortably
- Close other memory-intensive applications
- Consider using Kokoro or Piper for lower memory usage

## Links

- [Orpheus TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [orpheus-cpp GitHub](https://github.com/freddyaboulton/orpheus-cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## License

Orpheus TTS is released under the Apache 2.0 license.
