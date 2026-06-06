#!/usr/bin/env bash
# start.sh — bring up the Ayurvedic Leaf Detection app with one command.
#
# It is a single Streamlit web app (identify + search + browse), so "everything"
# is the startup chain around it:
#   1. move to the project root (this script's own directory)
#   2. use the Python 3.11 virtualenv in .venv
#   3. make sure dependencies are installed  (--install to force a (re)install)
#   4. sanity-check the trained model + knowledge base
#   5. launch the Streamlit UI
#
# Usage:
#   ./start.sh                       # start on http://localhost:8501
#   ./start.sh --install             # (re)install requirements first, then start
#   PORT=8600 ./start.sh             # choose a different port
#   HOST=0.0.0.0 ./start.sh          # expose on the LAN
#   HEADLESS=1 ./start.sh            # don't auto-open a browser tab
#   ./start.sh -- --foo bar          # pass extra args straight through to streamlit

set -euo pipefail

# --- locate project root so the script works from any directory -------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
HOST="${HOST:-localhost}"
PORT="${PORT:-8501}"

log()  { printf '\033[1;32m▶ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✖ %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

# --- parse flags ------------------------------------------------------------
INSTALL=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--install) INSTALL=1; shift ;;
    -h|--help)    usage ;;
    --)           shift; PASSTHRU+=("$@"); break ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

# --- 1. virtualenv (Python 3.11) -------------------------------------------
# The system python is 3.14 with no deps; everything runs out of .venv.
if [[ ! -x "$PY" ]]; then
  die "No virtualenv at .venv. Create it once with:
     python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi
log "Python: $("$PY" --version 2>&1)  (.venv)"
# PyTorch ships wheels for 3.11/3.12; a venv accidentally rebuilt on 3.13+/3.14
# fails to import torch. Warn (don't die) so a working newer build still runs.
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PYVER" in
  3.11|3.12) ;;
  *) warn "venv Python is $PYVER — torch wheels target 3.11/3.12; recreate with python3.11 if imports fail." ;;
esac

# --- 2. dependencies --------------------------------------------------------
if [[ "$INSTALL" -eq 1 ]]; then
  log "Installing requirements…"
  "$PY" -m pip install -q -r requirements.txt
elif ! "$PY" -c 'import streamlit, torch, sklearn' 2>/dev/null; then
  warn "Core dependencies missing — installing requirements (one-off)…"
  "$PY" -m pip install -q -r requirements.txt
fi

# --- 3. artifacts -----------------------------------------------------------
[[ -f data/knowledge_base/plants.json ]] \
  || die "Knowledge base missing: data/knowledge_base/plants.json"

if [[ -f models/best_model.pt ]]; then
  log "Trained model found: models/best_model.pt"
else
  warn "No models/best_model.pt — the 'Identify a leaf' tab will be disabled"
  warn "  (Search & Browse still work). Train with:  $PY -m src.train"
fi

# --- 4. launch --------------------------------------------------------------
# Skip Streamlit's first-run usage-stats / email prompt so startup never blocks.
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEADLESS_FLAG=()
[[ "${HEADLESS:-0}" == "1" ]] && HEADLESS_FLAG=(--server.headless true)

log "Starting Streamlit → http://$HOST:$PORT   (Ctrl-C to stop)"
# exec so Ctrl-C / signals reach Streamlit directly and the venv launcher
# isn't left as a dangling parent process.
exec "$PY" -m streamlit run app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  ${HEADLESS_FLAG[@]+"${HEADLESS_FLAG[@]}"} \
  ${PASSTHRU[@]+"${PASSTHRU[@]}"}
