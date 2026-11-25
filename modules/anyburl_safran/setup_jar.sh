#!/usr/bin/env bash
# Download the official AnyBURL jar next to the wrapper (gitignored).
# Override the version with ANYBURL_URL=... ./setup_jar.sh
set -euo pipefail
cd "$(dirname "$0")"
URL="${ANYBURL_URL:-https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-23-1x.jar}"
JAR="$(basename "$URL")"
if [ ! -f "$JAR" ]; then
  echo "Downloading $URL"
  curl -fL --retry 3 -o "$JAR" "$URL"
fi
ls -l "$JAR"
java -version 2>&1 | head -1 || echo "WARNING: no java on PATH; rule mining needs a JDK 11+."
echo "Jar ready: $(pwd)/$JAR"
