# Data package

Archived with the Zenodo record (not in git because of size):

| File | Content |
|---|---|
| `data_raw.zip` | `data/raw/`: EIA hourly demand, FRED-MD vintages 2025-04 to 2026-10, CDC FluView issue history, Wikipedia page views, M4 train/test CSVs |
| `forecasts.zip` | `results/forecasts/`: every model's forecasts per (origin, series) as parquet, with `.cost.json` timing/VRAM records |
| `SHA256SUMS.txt` | SHA-256 of both archives (verify with `sha256sum -c SHA256SUMS.txt`) |

Unzip both at the repository root.

## Sources and terms
| Dataset | Source | Terms |
|---|---|---|
| EIA hourly demand by balancing authority | U.S. Energy Information Administration open data API (`electricity/rto/region-data`) | U.S. Government public data |
| FRED-MD monthly vintages | McCracken and Ng, Federal Reserve Bank of St. Louis (research.stlouisfed.org/econ/mccracken/fred-databases) | Free for research use; cite McCracken and Ng (2016) |
| CDC FluView (ILINet) | U.S. CDC via the Delphi Epidata API | U.S. Government public data |
| Wikipedia daily page views | Wikimedia REST API | CC0 (Wikimedia pageview data) |
| M4 | Makridakis et al. (2020), github.com/Mcompetition/M4-methods | Free for research use; cite the M4 paper |

Snapshot dates: data fetched up to 2026-11-10 (EIA), CDC issue 2026-40, FRED-MD vintage 2026-10 (2026-08 vintage dropped, corrupted at source).
Live endpoints return later data than the snapshot; use the archive to reproduce the reported numbers.

## Model weights
Not redistributed. Models are downloaded from Hugging Face at run time (see `src/models/`). Qwen3-8B,
Mistral-7B-Instruct-v0.3, Chronos-Bolt, Chronos-2, TimesFM 2.5 and Moirai 2.0 keep their own licences.
