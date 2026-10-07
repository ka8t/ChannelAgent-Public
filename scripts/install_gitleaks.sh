#!/usr/bin/env bash
# Installs gitleaks, the secret scanner, into the virtualenv: .venv/bin/gitleaks.
#
#   scripts/install_gitleaks.sh            # into ./.venv
#   VENV_DIR=/path/to/venv scripts/install_gitleaks.sh
#
# gitleaks is a Go program with no pip package, so the official release archive is
# downloaded from github.com/gitleaks/gitleaks, checked against the SHA-256 pinned below
# (from the release's checksums file) and unpacked into the virtualenv. Nothing is
# installed when the pinned version is already there. A checksum that does not match
# installs nothing. start.sh --native runs it after the pip install.
set -euo pipefail

VERSION="8.30.1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
TARGET="$VENV_DIR/bin/gitleaks"

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)          PLATFORM="darwin_arm64"
                         SHA256="b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5" ;;
  Darwin/x86_64)         PLATFORM="darwin_x64"
                         SHA256="dfe101a4db2255fc85120ac7f3d25e4342c3c20cf749f2c20a18081af1952709" ;;
  Linux/aarch64|Linux/arm64)
                         PLATFORM="linux_arm64"
                         SHA256="e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080" ;;
  Linux/x86_64)          PLATFORM="linux_x64"
                         SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb" ;;
  *) echo "gitleaks: no pinned release for $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

if [ -x "$TARGET" ] && [ "$("$TARGET" version 2>/dev/null)" = "$VERSION" ]; then
  exit 0
fi
if [ ! -d "$VENV_DIR/bin" ]; then
  echo "gitleaks: no virtualenv at $VENV_DIR (create it first: python3 -m venv .venv)" >&2
  exit 1
fi

ARCHIVE="gitleaks_${VERSION}_${PLATFORM}.tar.gz"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "==> Installing gitleaks $VERSION into $VENV_DIR/bin"
curl -fsSL --proto '=https' -o "$WORK/$ARCHIVE" \
  "https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}/${ARCHIVE}"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$WORK/$ARCHIVE" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$WORK/$ARCHIVE" | cut -d' ' -f1)"
fi
if [ "$actual" != "$SHA256" ]; then
  echo "gitleaks: checksum mismatch for $ARCHIVE (expected $SHA256, got $actual): not installed" >&2
  exit 1
fi
tar -xzf "$WORK/$ARCHIVE" -C "$WORK" gitleaks
install -m 755 "$WORK/gitleaks" "$TARGET"
echo "==> gitleaks $("$TARGET" version) installed"
