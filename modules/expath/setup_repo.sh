#!/usr/bin/env bash
# Clone the eXpath reference implementation next to the wrapper (gitignored). Only needed to train the
# KGE base model (scripts/train_kge.py); serving does not import it.
# Optionally pin a commit: EXPATH_REF=<sha> ./setup_repo.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d expath-ref/.git ]; then
  git clone https://github.com/cs-anonymous/eXpath.git expath-ref
fi
if [ -n "${EXPATH_REF:-}" ]; then
  git -C expath-ref checkout "$EXPATH_REF"
fi
echo "eXpath at: $(git -C expath-ref rev-parse HEAD)  (tested with 2cb14a6)"
