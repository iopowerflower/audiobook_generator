# Audiobook Generator

A multi-engine text-to-speech application for generating high-quality audiobooks. Supports both cloud-based and offline TTS engines with a beautiful web interface.

## Features

- **Multiple TTS Engines**
  - **Google Chirp3-HD**: Cloud-based, premium voices with natural expression
  - **Kokoro**: Offline neural TTS with high-quality voices
  - **Piper**: Fast, lightweight offline TTS
  - **Orpheus**: LLM-based TTS with emotion support
  - **Docker backends**: Extensible support for containerized TTS engines (Kani, etc.)

- **Voice Preview & Selection**: Audition voices before generating your audiobook
- **Batch Processing**: Process entire manuscripts with progress tracking
- **Smart Caching**: Automatic preview caching for faster browsing
- **Pronunciation Overrides**: Custom pronunciation dictionary support
- **Modern UI**: Clean, accessible interface with dark theme

## Quick Start

### Prerequisites

- **Python 3.11** (required for all offline TTS engines)
- Git Bash (Windows) or any bash-compatible shell
- For GPU acceleration: NVIDIA GPU with CUDA support (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd audiobook_generator
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your Google API key (for cloud TTS)
   ```

3. **Run the setup script**

   For CPU-only setup:
   ```bash
   bash run_app.sh
   ```

   For GPU/CUDA support:
   ```bash
   bash run_app_cuda.sh
   ```

   The script will:
   - Create a Python 3.11 virtual environment
   - Install all dependencies
   - Set up offline TTS engines (Kokoro, Piper, Orpheus)
   - Start the Flask server

4. **Open your browser**
   ```
   http://localhost:5000
   ```

## Configuration

### Google Cloud TTS (Optional)

To use Google Chirp3-HD voices, you need a Google Cloud API key with Text-to-Speech API enabled:

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Cloud Text-to-Speech API
3. Create an API key
4. Add to `.env`:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Offline TTS Engines

All offline engines work without API keys or internet:

- **Kokoro**: See [README_kokoro.md](README_kokoro.md) for model download instructions
- **Piper**: See [README_piper.md](README_piper.md) for voice installation
- **Orpheus**: See [README_orpheus.md](README_orpheus.md) for setup and emotion tags

### Docker Backends (Advanced)

For containerized TTS engines like Kani:

- See [docker_backends/README.md](docker_backends/README.md) for setup instructions
- Requires Docker and NVIDIA Container Toolkit for GPU support

## Project Structure

```
audiobook_generator/
├── app.py                  # Main Flask application
├── templates/              # Web UI templates
├── requirements.txt        # Python dependencies
├── run_app.sh             # CPU setup script
├── run_app_cuda.sh        # GPU setup script
├── env.example            # Environment template
├── .gitignore             # Git ignore rules
├── pronunciations.json    # Custom pronunciation dictionary
├── models/                # Offline TTS model files (gitignored)
│   ├── kokoro/
│   └── piper/
├── cache/                 # Preview audio cache (gitignored)
├── docker_backends/       # Docker backend integrations
└── README_*.md           # Engine-specific documentation
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Cloud TTS API key | _(required for cloud voices)_ |
| `PIPER_MODELS_DIR` | Custom Piper models directory | `models/piper/` |
| `KOKORO_MODEL_PATH` | Custom Kokoro model path | `models/kokoro/kokoro-v1.0.fp16.onnx` |
| `PREVIEW_CACHE_DIR` | Voice preview cache location | `cache/previews/` |
| `PRONUNCIATIONS_PATH` | Custom pronunciations file | `pronunciations.json` |
| `ONNX_PROVIDER` | ONNX Runtime provider | _(auto-detect)_ |

See individual engine README files for more options.

## Usage

### Generating an Audiobook

1. Open http://localhost:5000
2. **Select a voice** from any TTS engine tab
3. **Preview voices** using the play button
4. **Choose files** to convert (txt, epub, pdf, docx)
5. **Adjust settings** (speed, chunk size)
6. **Click "Convert to Audiobook"**
7. **Download** individual files or complete ZIP

### Custom Pronunciations

Edit `pronunciations.json` to override default pronunciations:

```json
{
  "Hermione": "her-MY-oh-nee",
  "Leviosa": "leh-vee-OH-sah"
}
```

Changes apply immediately (no restart needed).

## Troubleshooting

### Python Version Issues
- Offline engines require **Python 3.11** (not 3.13+)
- Use `py -3.11` launcher or install from python.org

### Missing Dependencies
```bash
# Reinstall all dependencies
. .venv/Scripts/activate
pip install -r requirements.txt
```

### GPU/CUDA Issues
- See [README_kokoro.md](README_kokoro.md) for CUDA setup
- Try CPU fallback with `run_app.sh` if GPU fails

### API Errors
- Check `.env` has valid `GOOGLE_API_KEY`
- Verify API is enabled in Google Cloud Console
- Check quota limits

## Development

### Running Tests
```bash
. .venv/Scripts/activate
python -m pytest
```

### Adding a New TTS Backend
1. Create backend class in `docker_backends/`
2. Implement `interface.DockerTTSBackend`
3. Register in `docker_backends/manager.py`

See [docker_backends/README.md](docker_backends/README.md) for details.

## License

This project integrates multiple TTS engines, each with its own license:

- **Kokoro**: Check [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- **Piper**: Check [rhasspy/piper](https://github.com/rhasspy/piper) (MIT)
- **Orpheus**: Check [orpheus-cpp](https://github.com/freddyaboulton/orpheus-cpp)
- **Google Cloud TTS**: [Google Cloud Terms](https://cloud.google.com/terms)

Piper voices have individual licenses - check model cards before commercial use.

## Acknowledgments

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) by thewh1teagle
- [Piper TTS](https://github.com/rhasspy/piper) by Rhasspy
- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) by Canopy AI
- [orpheus-cpp](https://github.com/freddyaboulton/orpheus-cpp) by freddyaboulton
- Google Cloud Text-to-Speech API

## Support

For issues, questions, or contributions, please open an issue on GitHub.
