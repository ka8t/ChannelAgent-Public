#!/usr/bin/env bash
# Scans for secrets with the virtualenv's gitleaks (scripts/install_gitleaks.sh).
#
#   scripts/secret_scan.sh             # every commit of the git history
#   scripts/secret_scan.sh --dir PATH  # every file under PATH (gitleaks reads ignored files
#                                      # too: point it at a clean copy, never at a checkout
#                                      # holding .env, data/ or models/)
#
# The rules are gitleaks' defaults plus .gitleaks.toml (exact test values allowed, never
# whole files). Findings are printed redacted. Exit 0: no leak; 1: a leak; 2: no gitleaks.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GITLEAKS="${VENV_DIR:-$REPO_DIR/.venv}/bin/gitleaks"
if [ ! -x "$GITLEAKS" ]; then
  echo "gitleaks is not installed: run scripts/install_gitleaks.sh" >&2
  exit 2
fi
EXTRA=()
if [ -f "$REPO_DIR/.gitleaks-baseline.json" ]; then
  EXTRA=(--baseline-path "$REPO_DIR/.gitleaks-baseline.json")
fi
CONFIG="$REPO_DIR/.gitleaks.toml"
if [ "${1:-}" = "--dir" ]; then
  if [ -z "${2:-}" ] || [ ! -d "$2" ]; then
    echo "usage: scripts/secret_scan.sh --dir PATH" >&2
    exit 2
  fi
  exec "$GITLEAKS" dir "$2" --redact --no-banner --no-color --config "$CONFIG"
fi
cd "$REPO_DIR"
exec "$GITLEAKS" git . --redact --no-banner --no-color --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
