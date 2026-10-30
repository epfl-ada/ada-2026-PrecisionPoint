#!/usr/bin/env bash
# Download and extract the Wikispeedia dataset (SNAP) into data/raw/.
#
#   bash src/scripts/download_data.sh [target_folder] [--html]
#
# Fetches the paths/graph archive (9.5 MB) and the plaintext articles (35 MB)
# and checks them against the SHA-256 sums of the official release.
# The 755 MB HTML archive is only needed for exact link positions (planned
# for Milestone P3); pass --html as second argument to fetch it as well.
set -euo pipefail

RAW_DIR="${1:-data/raw}"
BASE_URL="https://snap.stanford.edu/data/wikispeedia"
mkdir -p "$RAW_DIR"

FILES=(wikispeedia_paths-and-graph.tar.gz wikispeedia_articles_plaintext.tar.gz)
if [[ "${2:-}" == "--html" ]]; then
  FILES+=(wikispeedia_articles_html.tar.gz)
fi

for f in "${FILES[@]}"; do
  if [[ ! -f "$RAW_DIR/$f" ]]; then
    echo "downloading $f"
    curl -L --fail -o "$RAW_DIR/$f" "$BASE_URL/$f"
  fi
done

cd "$RAW_DIR"

# Checksums are verified in Python because sha256sum options differ between Linux and macOS.
python3 - <<'PYCHECK'
import hashlib
import pathlib

expected = {
    "wikispeedia_paths-and-graph.tar.gz":
        "97697096f5d2dcb77aa69e3992305c6c561de89edb9fb10b5ad9feaf8ba534d5",
    "wikispeedia_articles_plaintext.tar.gz":
        "8e43d12822d05746a59f7de8987ffd6eafa57d248303d1c02cfe85b519359dc1",
}
for name, digest in expected.items():
    path = pathlib.Path(name)
    if path.exists():
        ok = hashlib.sha256(path.read_bytes()).hexdigest() == digest
        print(f"{name}: " + ("checksum OK" if ok else "WARNING, checksum differs from the official release"))
PYCHECK

for f in "${FILES[@]}"; do
  echo "extracting $f"
  tar -xzf "$f"
done
echo "done: $(ls)"
