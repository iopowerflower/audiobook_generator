#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Audiobook Generator - Setup & Run Script (Windows Git Bash)
# =============================================================================
#
# This script:
# - Ensures Python 3.11 is available
# - Creates/activates the virtual environment
# - Installs all dependencies (including optional TTS backends)
# - Starts the Flask app
#
# Usage:
#   cd audiobook_generator
#   bash run_app.sh
#
# Optional: Skip certain TTS backends
#   SKIP_ORPHEUS=1 bash run_app.sh    # Skip Orpheus TTS
#   SKIP_KOKORO=1 bash run_app.sh     # Skip Kokoro TTS
#   SKIP_PIPER=1 bash run_app.sh      # Skip Piper TTS
#
# For CUDA/GPU support, use run_app_cuda.sh instead.
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

if ! find_python311; then
  echo "ERROR: Python 3.11 not found."
  echo ""
  echo "Please install Python 3.11 from https://www.python.org/downloads/"
  echo "Make sure to check 'Add Python to PATH' and 'Install py launcher' during install."
  echo ""
  echo "If Python 3.11 is installed but not detected, try:"
  echo "  /c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv"
  exit 1
fi

echo "Found Python 3.11: $PYTHON_CMD"

# -----------------------------------------------------------------------------
# Virtual Environment Setup
# -----------------------------------------------------------------------------

venv_python_ok() {
  [[ -x ".venv/Scripts/python.exe" ]] || return 1
  ".venv/Scripts/python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" 2>/dev/null
}

if [[ ! -f ".venv/Scripts/activate" ]] || ! venv_python_ok; then
  if [[ -d ".venv" ]]; then
    echo "Venv exists but is not Python 3.11.x. Recreating .venv..."
    rm -rf ".venv"
  else
    echo "Creating virtual environment..."
  fi
  $PYTHON_CMD -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/Scripts/activate

PYTHON_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
if [[ ! -f "$PYTHON_EXE" ]]; then
  echo "ERROR: venv python not found at $PYTHON_EXE"
  exit 1
fi

echo "Using Python: $("$PYTHON_EXE" --version)"

# -----------------------------------------------------------------------------
# Install Base Dependencies
# -----------------------------------------------------------------------------

echo ""
echo "Installing base dependencies..."
"$PYTHON_EXE" -m pip install --upgrade pip --quiet 2>/dev/null || true
"$PYTHON_EXE" -m pip install -r requirements.txt --quiet 2>/dev/null || {
  echo "Installing dependencies (this may take a minute)..."
  "$PYTHON_EXE" -m pip install -r requirements.txt
}

# -----------------------------------------------------------------------------
# Install Orpheus TTS (optional, requires llama-cpp-python)
# -----------------------------------------------------------------------------

install_llama_cpp_python() {
  # Check if already installed
  if "$PYTHON_EXE" -c "import llama_cpp" 2>/dev/null; then
    return 0
  fi

  echo "  Installing llama-cpp-python..."

  # 1. Try official pre-built CPU wheels (hardware-agnostic, most portable)
  echo "    Trying official CPU wheel..."
  if "$PYTHON_EXE" -m pip install llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      --quiet 2>/dev/null; then
    echo "    llama-cpp-python (CPU) installed successfully"
    return 0
  fi

  # 2. Try default pip install (may trigger source build if compiler available)
  echo "    Trying default pip install..."
  if "$PYTHON_EXE" -m pip install llama-cpp-python --quiet 2>/dev/null; then
    echo "    llama-cpp-python installed successfully"
    return 0
  fi

  echo ""
  echo "  WARNING: llama-cpp-python installation failed."
  echo "           Orpheus TTS voices will not work."
  echo ""
  echo "  Manual install options:"
  echo "    1. Install Visual Studio Build Tools, then: pip install llama-cpp-python"
  echo "    2. Download a pre-built wheel from: https://github.com/dougeeai/llama-cpp-python-wheels/releases"
  echo "       Then: pip install <downloaded-wheel.whl>"
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
    echo ""
    echo "Installing Orpheus TTS (LLM-based, high-quality voices)..."
    "$PYTHON_EXE" -m pip install orpheus-cpp --quiet 2>/dev/null || {
      echo "  Installing orpheus-cpp..."
      "$PYTHON_EXE" -m pip install orpheus-cpp
    }
  fi

  # ALWAYS check llama-cpp-python (orpheus-cpp can install without it)
  install_llama_cpp_python

  # Verify installation
  if "$PYTHON_EXE" -c "import llama_cpp" 2>/dev/null; then
    echo "  Orpheus TTS: ready"
  fi
}

install_orpheus

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  AUDIOBOOK GENERATOR"
echo "=============================================="
echo ""
echo "Python: $("$PYTHON_EXE" -c 'import sys; print(sys.version.split()[0])')"

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
else
  echo "Offline TTS: (none installed)"
fi

echo "Cloud TTS: Google Chirp3-HD"
echo ""
echo "Open http://localhost:5000 in your browser"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Start the App
# -----------------------------------------------------------------------------

"$PYTHON_EXE" app.py
