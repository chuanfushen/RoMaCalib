# Ablation protocols

## Table 4 — SAM foreground mask

The paper comparison is paired, not a pair of unrelated samples:

- `robot_in_view`: 900 requested frames, composed of 300 full-body and 600
  partial-view frames;
- `robot_in_and_out`: 900 requested frames;
- both conditions use six views and three refinement iterations;
- Stage 0 matches the six orbit renders with `match_batch_size = 6`; each
  R1--R3 pose-aligned replay has one render and uses the historical one-pair
  matcher call (`refinement_match_batch_size = 1`);
- the only experimental variable is `mask_input`.

`SamAblationPair` rejects changed protocol settings and requires the complete
no-SAM/SAM manifest SHA-256 values to match (not merely their frame IDs).  The
validator prints both hashes and their shared value.  The public runtime
boundary is `calibx.benchmarks.ablations.sam_executor`: it keeps Stage 0 and
R1--R3 as different typed requests, records the frozen gate-v1 seed and only
portable artifact IDs, and emits explicit T0--R3 unavailable rows when Stage
0 has no PnP pose.  For an available T0 pose it uses the gate-v1 trajectory,
including reject/small-update/two-cycle filling semantics.

`scripts/reproduce/ablations/run_sam_table4.py` defaults to an asset-free
four-arm plan.  `--validate-manifests` additionally enforces the two paired
900-frame manifests.  `--execute` deliberately requires a caller-supplied
local typed backend; the public tree does not guess a private Panda/DROID
reader, checkpoint, output path, or paper number.  A CF golden-output
regression is still required before a new run may be labelled a paper result.

## Table 5 — feature matcher

The candidate protocol fixes 300 Panda-ORB frames, six views, three
iterations, `gate-v1`, and a SAM input mask.  The audited launcher divided the
same parent selection into two 150-frame shards.  Every matcher, including
the RoMaV2 baseline, must record the same parent manifest SHA-256 and the
same two child-shard SHA-256 values, then merge per-frame records before
computing aggregate metrics.

Its external methods require separately reviewed installations; they are not
a bundled `pyproject.toml` extra.  The currently recovered formal outputs do
**not** equal the paper's Table 5 values, so this module defines a comparison
contract only; it does not publish or claim those values.  No host path,
checkpoint, private dependency tree, historical hash, or output is included
in the public configuration.

`calibx.benchmarks.ablations.external_matchers` contains lazy wrappers for
the three external methods.  Importing the module does not import or install
their projects.  RoMa v1 and MASt3R require caller-supplied local checkpoints;
LightGlue requires an explicit provider opt-in.  This keeps optional software
outside the normal CLI/help/import path and prevents the source tree from
silently downloading an experiment dependency.
