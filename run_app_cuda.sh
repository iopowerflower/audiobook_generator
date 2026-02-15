#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Audiobook Generator - CUDA/GPU Setup & Run Script (Windows Git Bash)
# =============================================================================
#
# What it does:
# - Ensures Python 3.11 is available
# - Creates/activates the virtual environment
# - Installs all dependencies (including Orpheus TTS with llama-cpp-python)
# - Prepends cuDNN (CUDA 12) bin dir to PATH so cudnn64_9.dll is discoverable
# - Forces ONNX Runtime to use CUDAExecutionProvider
# - Starts the Flask app
#
# Usage:
#   cd audiobook_generator
#   bash run_app_cuda.sh
#
# Optional overrides:
#   CUDNN_BIN="/c/Program Files/NVIDIA/CUDNN/v9.17/bin/12.9" bash run_app_cuda.sh
#   ONNX_PROVIDER=CUDAExecutionProvider bash run_app_cuda.sh
#   SKIP_ORPHEUS=1 bash run_app_cuda.sh   # Skip Orpheus TTS installation
# =============================================================================

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# Python 3.11 Detection (supports both py launcher and direct path)
# -----------------------------------------------------------------------------

PYTHON_CMD=""

find_python311() {
  # Try py launcher first (standard Windows)
  if command -v py >/dev/null 2>&1; then
    if py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" 2>/dev/null; then
      PYTHON_CMD="py -3.11"
      return 0
    fi
  fi

  # Try direct path (common install locations)
  local candidates=(
    "/c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe"
    "/c/Python311/python.exe"
    "/c/Program Files/Python311/python.exe"
    "python3.11"
    "python"
  )

  for cmd in "${candidates[@]}"; do
    if [[ -x "$cmd" ]] || command -v "$cmd" >/dev/null 2>&1; then
      if $cmd -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" 2>/dev/null; then
        PYTHON_CMD="$cmd"
        return 0
      fi
    fi
  done

  return 1
}

need_python311() {
  find_python311
}

ensure_python311() {
  if find_python311; then
    echo "Found Python 3.11: $PYTHON_CMD"
    return 0
  fi

  echo "ERROR: Python 3.11 not found."
  echo ""
  echo "Please install Python 3.11 from https://www.python.org/downloads/"
  echo "Make sure to check 'Add Python to PATH' and 'Install py launcher' during install."
  echo ""
  echo "If Python 3.11 is installed but not detected, try:"
  echo "  /c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv"
  exit 1
}

venv_python_ok() {
  # Returns 0 if venv python exists and is 3.11.x
  [[ -x ".venv/Scripts/python.exe" ]] || return 1
  ".venv/Scripts/python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >/dev/null 2>&1
}

ensure_python311

if [[ ! -f ".venv/Scripts/activate" ]] || ! venv_python_ok; then
  if [[ -d ".venv" ]]; then
    echo "Venv exists but is not Python 3.11.x. Recreating .venv..."
    rm -rf ".venv"
  else
    echo "Creating venv at $ROOT_DIR/.venv using Python 3.11..."
  fi
  $PYTHON_CMD -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/Scripts/activate

PYTHON_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
if [[ ! -f "$PYTHON_EXE" ]]; then
  echo "ERROR: venv python not found at $PYTHON_EXE"
  echo "       Ensure the venv was created successfully, then re-run this script."
  exit 1
fi

# -----------------------------------------------------------------------------
# Install Dependencies
# -----------------------------------------------------------------------------

# Check if base deps are already installed (skip slow pip check on every run)
if ! "$PYTHON_EXE" -c "import flask, kokoro_onnx" 2>/dev/null; then
  echo "Installing dependencies..."
  "$PYTHON_EXE" -m pip install --upgrade pip --quiet 2>/dev/null || true
  "$PYTHON_EXE" -m pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "Installing dependencies (this may take a minute)..."
    "$PYTHON_EXE" -m pip install -r requirements.txt
  }
fi

# -----------------------------------------------------------------------------
# Ensure GPU ONNX Runtime (replace CPU version if needed)
# -----------------------------------------------------------------------------

ensure_onnxruntime_gpu() {
  # Check if CUDAExecutionProvider is already available and working
  if "$PYTHON_EXE" -c "import onnxruntime as rt; exit(0 if 'CUDAExecutionProvider' in rt.get_available_providers() else 1)" 2>/dev/null; then
    return 0
  fi

  echo "Setting up onnxruntime-gpu..."
  
  # Check if onnxruntime-gpu is installed but not working (vs CPU version installed)
  local current_pkg
  current_pkg=$("$PYTHON_EXE" -m pip show onnxruntime-gpu 2>/dev/null | grep -c "Name: onnxruntime-gpu" || echo "0")
  
  if [[ "$current_pkg" == "0" ]]; then
    # onnxruntime-gpu not installed, need to replace CPU version
    echo "  Replacing onnxruntime (CPU) with onnxruntime-gpu..."
    "$PYTHON_EXE" -m pip uninstall -y onnxruntime onnxruntime-directml 2>/dev/null || true
    "$PYTHON_EXE" -m pip install onnxruntime-gpu --quiet || {
      echo "  Installing onnxruntime-gpu..."
      "$PYTHON_EXE" -m pip install onnxruntime-gpu
    }
  fi
  
  # Verify it works
  if ! "$PYTHON_EXE" -c "import onnxruntime as rt; exit(0 if 'CUDAExecutionProvider' in rt.get_available_providers() else 1)" 2>/dev/null; then
    echo "WARNING: CUDAExecutionProvider not available. GPU acceleration disabled."
    echo "         This may be due to missing CUDA drivers or cuDNN."
    return 1
  fi
}

ensure_onnxruntime_gpu

# -----------------------------------------------------------------------------
# Install Orpheus TTS (optional)
# -----------------------------------------------------------------------------

install_llama_cpp_python() {
  # Check if already installed AND working (import test catches DLL issues)
  if "$PYTHON_EXE" -c "import llama_cpp" 2>/dev/null; then
    return 0
  fi
  
  # Clean up any broken installs
  "$PYTHON_EXE" -m pip uninstall -y llama-cpp-python 2>/dev/null || true

  echo "Installing llama-cpp-python..."

  # Strategy: Build from source first (most reliable), then fall back to pre-built wheels
  # Pre-built wheels often have CPU instruction compatibility issues (AVX-512 vs AVX2)

  # 1. Try building from source (if C++ compiler available - most compatible)
  if command -v cl.exe >/dev/null 2>&1 || [[ -n "${VSCMD_VER:-}" ]]; then
    echo "  Building from source (C++ compiler detected)..."
    if "$PYTHON_EXE" -m pip install llama-cpp-python --no-cache-dir --quiet 2>/dev/null; then
      if "$PYTHON_EXE" -c "import llama_cpp" 2>/dev/null; then
        echo "  llama-cpp-python built from source successfully"
        return 0
      fi
    fi
  fi

  # 2. Try official pre-built CPU wheels
  echo "  Trying official CPU wheel..."
  if "$PYTHON_EXE" -m pip install llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      --quiet 2>/dev/null; then
    if "$PYTHON_EXE" -c "import llama_cpp" 2>/dev/null; then
      echo "  llama-cpp-python (CPU) installed successfully"
      return 0
    else
      echo "  CPU wheel installed but failed import test, removing..."
      "$PYTHON_EXE" -m pip uninstall -y llama-cpp-python 2>/dev/null || true
    fi
  fi

  # 3. Try default pip install (may build from source or use PyPI wheel)
  echo "  Trying default pip install..."
  if "$PYTHON_EXE" -m pip install llama-cpp-python --no-cache-dir --quiet 2>/dev/null; then
    if "$PYTHON_EXE" -c "import llama_cpp" 2>/dev/null; then
      echo "  llama-cpp-python installed successfully"
      return 0
    else
      echo "  Install succeeded but import failed (DLL compatibility issue)"
      "$PYTHON_EXE" -m pip uninstall -y llama-cpp-python 2>/dev/null || true
    fi
  fi

  echo ""
  echo "WARNING: llama-cpp-python installation failed."
  echo "         Orpheus TTS voices will not work."
  echo ""
  echo "To fix, open 'Developer Command Prompt for VS' and run:"
  echo "  cd $(pwd)"
  echo "  .venv\\Scripts\\python.exe -m pip install llama-cpp-python --no-cache-dir"
  echo ""
  return 1
}

install_orpheus() {
  if [[ "${SKIP_ORPHEUS:-}" == "1" ]]; then
    echo "Skipping Orpheus TTS (SKIP_ORPHEUS=1)"
    return 0
  fi

  # Install orpheus-cpp if not present
  if ! "$PYTHON_EXE" -c "import orpheus_cpp" 2>/dev/null; then
    echo "Installing Orpheus TTS..."
    "$PYTHON_EXE" -m pip install orpheus-cpp --quiet 2>/dev/null || \
      "$PYTHON_EXE" -m pip install orpheus-cpp
  fi

  # ALWAYS check llama-cpp-python (orpheus-cpp can install without it)
  install_llama_cpp_python
}

install_orpheus

pick_cudnn_bin() {
  # Prefer user override if it exists
  if [[ -n "${CUDNN_BIN:-}" && -d "${CUDNN_BIN}" ]]; then
    echo "${CUDNN_BIN}"
    return 0
  fi

  # Auto-detect: newest v*/bin/12.* folder (works with NVIDIA cuDNN installer layout)
  local base="/c/Program Files/NVIDIA/CUDNN"
  local candidates=()

  # Note: glob may not match; avoid failing due to set -u/-e
  while IFS= read -r d; do
    candidates+=("$d")
  done < <(ls -d "$base"/v*/bin/12.* 2>/dev/null || true)

  if [[ ${#candidates[@]} -gt 0 ]]; then
    # Sort versions naturally and pick the last one
    printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
    return 0
  fi

  return 1
}

if CUDNN_BIN_DETECTED="$(pick_cudnn_bin)"; then
  export PATH="$CUDNN_BIN_DETECTED:$PATH"
else
  echo "WARN: Could not auto-detect cuDNN CUDA-12 bin directory."
  echo "      If you hit missing cudnn64_9.dll errors, set CUDNN_BIN manually, e.g.:"
  echo "      CUDNN_BIN=\"/c/Program Files/NVIDIA/CUDNN/v9.17/bin/12.9\" bash run_app_cuda.sh"
fi

export ONNX_PROVIDER="${ONNX_PROVIDER:-CUDAExecutionProvider}"

# -----------------------------------------------------------------------------
# Status Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  AUDIOBOOK GENERATOR (CUDA)"
echo "=============================================="
echo ""
echo "Python: $("$PYTHON_EXE" -c 'import sys; print(sys.version.split()[0])')"
echo "ONNX_PROVIDER: $ONNX_PROVIDER"
echo "cuDNN bin: ${CUDNN_BIN_DETECTED:-"(not detected)"}"

ORT_PROVIDERS="$("$PYTHON_EXE" -c 'import onnxruntime as rt; print(",".join(rt.get_available_providers()))' 2>/dev/null || true)"
echo "ONNX providers: ${ORT_PROVIDERS:-"(not importable)"}"

# Check available TTS backends
BACKENDS=""
if "$PYTHON_EXE" -c "from kokoro_onnx import Kokoro" 2>/dev/null; then
  BACKENDS="$BACKENDS Kokoro"
fi
if "$PYTHON_EXE" -c "import piper" 2>/dev/null; then
  BACKENDS="$BACKENDS Piper"
fi
if "$PYTHON_EXE" -c "from orpheus_cpp import OrpheusCpp" 2>/dev/null; then
  BACKENDS="$BACKENDS Orpheus"
fi
if [[ -n "$BACKENDS" ]]; then
  echo "Offline TTS:$BACKENDS"
fi

echo ""
echo "Open http://localhost:5000 in your browser"
echo "=============================================="
echo ""

if [[ "${ORT_PROVIDERS}" != *"CUDAExecutionProvider"* ]]; then
  echo "ERROR: CUDAExecutionProvider is not available in this Python env."
  echo "       Fix: uninstall CPU ORT and install GPU ORT, e.g.:"
  echo "       python -m pip uninstall -y onnxruntime"
  echo "       python -m pip install --upgrade --no-cache-dir onnxruntime-gpu"
  exit 1
fi

"$PYTHON_EXE" app.py

