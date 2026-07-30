# Calib-X

Calib-X is a training-free camera-to-robot pose-estimation pipeline based on
dense real-to-rendered feature matching, robot mesh correspondences, and
PnP-RANSAC.

This repository is being migrated from a clean RoMaV2 base plus the verified
historical CF/SZ experiment layers.  It now has the core Calib-X package
boundary plus paper protocol, reporting, and qualitative-selection contracts.
The public execution adapters preserve historical control flow while factoring
private paths, datasets, and outputs into local adapters; a planning or
validation command must not be mistaken for a completed full-GPU reproduction.

## Structure

```text
src/romav2/                 upstream dense matcher
src/calibx/                 calibration pipeline
src/calibx/dataset_adapters dataset readers, not dataset files
src/calibx/benchmarks/      paper protocol and manifest contracts
src/calibx/reporting/       paper metrics, provenance, tables, selections
config/                     shared TOML configuration
config/reproduction/        host-neutral paper-protocol templates
tests/                      fast offline correctness checks
scripts/benchmarks/         upstream matcher throughput/benchmark scripts
scripts/reproduce/          paper protocol planning and validation
scripts/paper/              compact-record table and qualitative-grid builders
paper/                      public manifests and generated-source locations
docs/                       architecture and reproduction notes
```

See [docs/architecture.md](docs/architecture.md) for module responsibilities
and compatibility policy.

## Installation

Python 3.12 is used for the validated experiment environment.

```bash
uv sync --frozen --extra dev --extra eval --extra setup
```

Datasets, pretrained weights, caches, and experiment outputs are not
distributed with the source tree. Machine-specific locations belong in an
ignored `config/dream_eval.local.toml` or the documented environment
variables. SAM3 masking uses the configured `sam3_checkpoint`; its BPE
tokenizer follows the historical SAM3 package behavior.

## Command line

Configuration and command generation can be checked without loading data or
models:

```bash
calibx run --config config/dream_eval.toml --dry-run
```

Recompute aggregate metrics from saved per-frame records:

```bash
calibx summarize outputs/<run-directory>
```

Aggregate top-level d7/Table 4 records with the paper's all-requested-frame
continuous AUC contract:

```bash
calibx paper-summarize outputs/<run-directory> --requested-frames <N>
```

`paper-summarize` is a reporting command only.  It does not mean every paper
protocol already has a public execution runner.

The historical entry point remains available during migration:

```bash
python scripts/run_dream_eval.py --config config/dream_eval.toml --dry-run
```

## Development checks

```bash
ruff check src tests
pytest -q
uv build
```

GPU throughput scripts and full benchmark runs are intentionally not part of
the default test suite.

## Upstream

The dense matcher under `src/romav2` is derived from RoMaV2. Its original
license and attribution are retained in this repository. Calib-X-specific
calibration code lives under `src/calibx`.
