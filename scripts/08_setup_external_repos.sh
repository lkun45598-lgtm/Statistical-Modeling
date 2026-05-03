#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-external}"
mkdir -p "$ROOT"

if [ ! -d "$ROOT/OpenSTL" ]; then
  git clone https://github.com/chengtan9907/OpenSTL "$ROOT/OpenSTL"
fi

if [ ! -d "$ROOT/CAS-Canglong" ]; then
  git clone https://github.com/GISWLH/CAS-Canglong "$ROOT/CAS-Canglong"
fi

if [ ! -d "$ROOT/Time-Series-Library" ]; then
  git clone https://github.com/thuml/Time-Series-Library "$ROOT/Time-Series-Library"
fi

echo "[ok] external repos cloned under $ROOT"
