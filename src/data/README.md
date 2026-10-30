# Data

Not versioned (see `.gitignore`). From the repository root:

* `bash src/scripts/download_data.sh` downloads and extracts the SNAP Wikispeedia archives into `data/raw/`
  (`wikispeedia_paths-and-graph/` and `plaintext_articles/`, 45 MB compressed).
* `python -m src.scripts.build_dataset` writes the cleaned tables (parquet) and arrays (`.npy`) to
  `data/processed/` (about 1.2 GB, most of it the 23.4M x 10 choice feature matrix).
