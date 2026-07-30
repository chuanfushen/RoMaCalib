# Architecture

Calib-X is separated from the upstream RoMaV2 matcher so that dataset
handling, calibration, evaluation, and presentation do not accumulate in one
benchmark utility module.

## Package boundary

- `romav2` owns dense feature matching.
- `calibx` owns camera-to-robot calibration.
- `calibx.benchmarks` owns dataset-specific paper protocol, manifest, and
  historical-semantic contracts; it does not replace the core pipeline.
- `calibx.reporting` owns paper metrics, public provenance, tables, and
  qualitative selection records.
- `config` contains editable TOML files; `calibx.configuration` contains the
  Python code that reads them.
- `tests` contains fast, deterministic correctness checks.
- Upstream GPU throughput and matcher benchmarks live in `scripts/benchmarks`.
- `scripts/reproduce` contains paper-protocol planning/validation while the
  historical execution adapters are reconstructed; `scripts/paper` consumes
  compact records to generate table source and qualitative grids.

The former `romav2.dream_config` and
`romav2.benchmarks.utils.{evaluator,geometry,mask,pose,prerender,render}`
modules remain as compatibility wrappers. New code must import from
`calibx`.

## Runtime responsibilities

```text
configuration
    ↓
dataset adapter
    ↓
pipeline ──→ masking
    │       matching
    │       rendering
    │       geometry
    │       pose
    ↓
frame result
    ├──→ metrics
    └──→ visualization
```

`calibx.pipeline` processes one frame. It does not select a paper subset or
aggregate a benchmark.

`calibx.runner` discovers frames, applies a manifest or deterministic sample,
isolates frame failures, and saves results.

`calibx.metrics` consumes saved frame records. Failed pose estimates contribute
to failure counts and success rate; mean, median, AUC, and threshold metrics
are computed from successful pose estimates only.

This upstream-compatible behavior is intentionally separate from
`calibx.reporting.paper_metrics`.  The latter implements d7/Table 4's
all-requested-frame continuous AUC contract; the two aggregators must not be
mixed when reproducing a paper table.

## Commands

```bash
calibx run --config config/dream_eval.toml --dry-run
calibx summarize outputs/<run>
calibx paper-summarize outputs/<run> --requested-frames <N>
```

Qualitative grids and compact table source are generated through
`scripts/paper/`, rather than a catch-all CLI command.  The reproduction
planning commands remain separate because they intentionally do not execute
historical CF workloads yet.

## Dependency rule

Importing `calibx.cli` or running `calibx --help` must not load model weights,
initialize CUDA, create a MuJoCo context, or download assets. Optional systems
such as SAM3 are imported only when the corresponding runtime feature is
enabled. The main-compatible caller interface supplies its checkpoint from the
configuration and retains SAM3's historical BPE-resolution behavior; it does
not restore the original developer-machine absolute asset paths.

## Migration rule

The refactor is behavior-preserving unless a change is explicitly documented
and tested. Compatibility wrappers may be removed only after downstream users
have migrated to the `calibx` namespace.
