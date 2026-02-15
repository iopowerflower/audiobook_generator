# Kokoro (Offline Neural TTS) Setup

Your current Windows install is **Python 3.13**, but **Kokoro TTS requires Python >=3.9,<3.13** (i.e. **Python 3.11 / 3.12**, not 3.13).

## 1) Install Python 3.11 (recommended)

Python 3.11 is the easiest choice on Windows because it’s widely available with normal installers and is supported by Kokoro.

Quick check (optional):

```bash
py -0p
```

You should see a `-V:3.11 ...` entry.

## 2) Create and activate venv (Windows / Git Bash)

From `audiobook_generator/`:

```bash
py -3.11 -m venv .venv

# IMPORTANT (Git Bash): activate must be sourced (dot-space) so it affects your current shell
. .venv/Scripts/activate

python -V  # should be 3.11.x
python -m pip install -r requirements.txt
```

If `python -V` still shows **3.13**, your venv is not activated. In Git Bash:

```bash
which python
python -c "import sys; print(sys.executable)"
```

## 3) Download Kokoro model files

Create `audiobook_generator/models/kokoro/` and download:

- `kokoro-v1.0.fp16.onnx` (preferred; the app will use this automatically if present)
- `kokoro-v1.0.onnx` (fallback)
- `voices-v1.0.bin`

Recommended (for `kokoro_onnx`): use the model files from `thewh1teagle/kokoro-onnx` releases (they are intended for ONNX Runtime execution providers like DirectML/CUDA).

Example (Git Bash):

```bash
mkdir -p audiobook_generator/models/kokoro

curl -L -o audiobook_generator/models/kokoro/kokoro-v1.0.fp16.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx

curl -L -o audiobook_generator/models/kokoro/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

curl -L -o audiobook_generator/models/kokoro/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

ls -la audiobook_generator/models/kokoro
```

If you already downloaded from `nazdridoy/kokoro-tts`, that can still work, but DirectML has been observed to fail on some systems/models.
mkdir -p audiobook_generator/models/kokoro

curl -L -o audiobook_generator/models/kokoro/kokoro-v1.0.onnx \
  https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx

curl -L -o audiobook_generator/models/kokoro/voices-v1.0.bin \
  https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

ls -la audiobook_generator/models/kokoro
Expected paths:

- `audiobook_generator/models/kokoro/kokoro-v1.0.fp16.onnx` (preferred)
- `audiobook_generator/models/kokoro/kokoro-v1.0.onnx` (fallback)
- `audiobook_generator/models/kokoro/voices-v1.0.bin`

Or set env vars:

- `KOKORO_MODEL_PATH`
- `KOKORO_VOICES_PATH`

## 4) Run the app

```bash
python app.py
```

If Kokoro is installed and the model files exist, you’ll see a **🎛️ Kokoro (Offline Neural)** section in the voice picker.

## GPU (optional)

Kokoro runs on **CPU by default** unless ONNX Runtime has a GPU execution provider available.

### Check available providers

In the same venv you run the app from:

```bash
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
```

If you see `CUDAExecutionProvider`, you can use NVIDIA GPU.

### Enable NVIDIA GPU (CUDA)

First, check what you currently have:

```bash
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
```

If you only see `CPUExecutionProvider` (and maybe `AzureExecutionProvider`), you are on CPU.

#### Recommended on Windows: DirectML (uses your GPU, avoids CUDA/cuDNN DLL headaches)

If you’re seeing errors like missing `cublasLt64_12.dll` / `cublas64_12.dll` / `cudnn64_9.dll`, your ONNX Runtime CUDA provider is installed but **cannot initialize**, so it falls back to CPU.

DirectML avoids that whole dependency chain on Windows.

In your venv:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -U onnxruntime-directml
export ONNX_PROVIDER=DmlExecutionProvider
python app.py
```

> On Windows PowerShell the env var is: `$env:ONNX_PROVIDER="DmlExecutionProvider"`

Confirm it’s active:

- Open `http://127.0.0.1:5000/api/diagnostics`
- Check `kokoro_session_active_providers` includes `DmlExecutionProvider`

If you see DirectML kernel errors (e.g. `ConvTranspose ... 80070057`), the app will automatically fall back to CPU for affected chunks. In that case, for better GPU reliability, try the `kokoro-onnx` model files above, or use CUDA (Option A).

#### Option A: CUDA (best if you want to manage CUDA 12 + cuDNN 9)

##### Install CUDA 12 and cuDNN 9 (Windows)

For NVIDIA GPU acceleration via ONNX Runtime CUDA, you need:

- CUDA 12.x (Toolkit or runtime)
- cuDNN 9 (for CUDA 12)

Quick verification in **PowerShell**:

```powershell
where.exe cublasLt64_12.dll
where.exe cublas64_12.dll
where.exe cudnn64_9.dll
```

If `cudnn64_9.dll` is **not found**, install cuDNN 9 for CUDA 12 and ensure its `bin` folder is on PATH (or copy the DLL into your CUDA `bin` folder, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`).

> Important: `python -c "import onnxruntime as rt; print(rt.get_available_providers())"` can show `CUDAExecutionProvider` even when CUDA can’t actually initialize due to missing DLLs. The definitive test is whether a CUDA session can be created (see below).

Uninstall CPU ONNX Runtime (important), then install GPU ONNX Runtime in the same venv:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install -U onnxruntime-gpu
```

Then force the provider (recommended):

```bash
export ONNX_PROVIDER=CUDAExecutionProvider
python app.py
```

> On Windows PowerShell the env var is: `$env:ONNX_PROVIDER="CUDAExecutionProvider"`

Re-check providers:

```bash
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
```

You want to see `CUDAExecutionProvider`.

Definitive CUDA session test (in the same venv):

```bash
python - <<'PY'
import onnxruntime as rt
m = "models/kokoro/kokoro-v1.0.fp16.onnx"
sess = rt.InferenceSession(m, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ACTIVE PROVIDERS:", sess.get_providers())
PY
```

You want `ACTIVE PROVIDERS:` to include `CUDAExecutionProvider`.

If you see errors about missing DLLs like `cublasLt64_12.dll` / `cublas64_12.dll` / `cudnn64_9.dll`, that means the CUDA 12 + cuDNN 9 runtime libraries are not discoverable on PATH (or you have a mismatch). In that case:

- Install an NVIDIA CUDA 12 runtime/toolkit (so those DLLs exist and are discoverable), **or**
- Use **DirectML** (recommended above), which does not require the CUDA toolkit.


