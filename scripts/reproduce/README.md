# Reproduction runners

This directory contains public planning and validation entry points migrated
from the verified CF protocols.  They preserve the historical control flow and
parameters; only private path/data/result handling is factored into local
adapters.  They invoke reusable modules in
`calibx.benchmarks` and do not embed server paths, data files, or paper
numbers.  The historical DREAM and CTRNet-X batch control layers now live in
`calibx.benchmarks`.  CTRNet-X batch exposes a typed runtime adapter in
`calibx.benchmarks.ctrnetx.batch_executor`; its dataset-specific hooks remain
deliberately separate from `calibx.runner` until a CF golden-output regression
proves the integration.  This is not an independent replacement algorithm.

CTRNet-X Panda single-frame has its own typed runtime boundary.  Validate a
complete portable frame manifest and print its plan with:

```powershell
python scripts/reproduce/ctrnetx/run_single_legacy.py `
  --config config/reproduction/ctrnetx/single.example.toml `
  --runtime-frame-manifest <portable-frame-manifest.json>
```

Add `--execute --hook-factory module:build_hooks` only on a server where that
local hook resolves the private assets.  The executor fixes R0 at six
views/batch six and applies the old unconditional-replace R1--R3 state
machine; it never substitutes the closed-loop batch or gate path.

For the historical full DREAM-real refinement, use the separate artifact-replay
entry rather than `calibx run`:

```powershell
python scripts/reproduce/dream_real/run_legacy_full.py `
  --config config/reproduction/dream_real/full.example.toml
```

That command only prints the four-run plan.  Add `--execute` only on a server
after its local configuration points `initial_results_root` at the matching
Stage-1 outputs.  It reuses `input_mask.png` if present, otherwise falls back
to the original RGB; it never reruns SAM or the Stage-1 orbit search.
For a resumable run, pass the same local `--runtime-identity-file` on the
initial and resume invocations.  It must identify the approved code, weights,
and environment; the executor additionally checks content digests of every
selected input and archived Stage-1 artifact before accepting resume.

For Table 4, the public entry is deliberately separate because its d7
gate-v1 refinement is not the archived DREAM legacy flow:

```powershell
python scripts/reproduce/ablations/run_sam_table4.py `
  --config config/reproduction/ablations/sam_table4.example.toml
```

This default command prints the two splits × two mask arms without reading
data.  Add `--validate-manifests` after filling the public 900-frame manifests.
Add `--execute --backend your_local_module:make_backend` only on a server with
a local typed backend that resolves opaque frame IDs; no built-in command
guesses dataset paths or downloads optional assets.
