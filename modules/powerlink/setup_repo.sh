#!/usr/bin/env bash
# Clone the PowerLink reference implementation next to the wrapper (gitignored).
# Optionally pin a commit: POWERLINK_REF=<sha> ./setup_repo.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d power-link/.git ]; then
  git clone https://github.com/Jiangnan0522/power-link.git power-link
fi
if [ -n "${POWERLINK_REF:-}" ]; then
  git -C power-link checkout "$POWERLINK_REF"
fi
echo "PowerLink at: $(git -C power-link rev-parse HEAD)  (record this SHA in the README)"
