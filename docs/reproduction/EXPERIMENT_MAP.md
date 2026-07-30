# Paper experiment map

The repository distinguishes a historical result from a public reproduction.
Only a clean commit, frozen configuration, public frame manifest, and matching
per-frame summary may be labelled reproducible.

| Paper item | Public protocol location | Requested scope | Current migration status |
| --- | --- | ---: | --- |
| DREAM-real full | `calibx.benchmarks.dream_real` | 49,619 frames across Azure, Kinect360, RealSense, and ORB | Four-camera plan, Stage-1 artifact reader, R0--R3 unconditional-replace state machine, and a lazy MuJoCo/RoMaV2 artifact-replay backend are present; CF remote golden-output regression remains pending |
| Panda single-frame | `calibx.benchmarks.ctrnetx.single_executor` | 17,306 frames: full-body 1,864, partial 6,264, in-and-out 9,178 | Three-split plan plus portable runtime manifest, typed Stage-0/refinement hooks, and the historical unconditional R0--R3 state machine are present; a local CTRNet-X asset hook and CF golden-output regression remain pending |
| Panda closed-loop batch | `calibx.benchmarks.ctrnetx` | 17,306 frames / 60 episodes | Manifest/shard planning, episode R0--R3 shared-pose state machine, all-frame coverage, cache-recovery audit, typed runtime frame/hook boundary, and optional MuJoCo/core bridges are present; the dataset adapter and CF remote golden-output regression remain pending |
| SAM foreground ablation | `calibx.benchmarks.ablations` | in-view 900 and in-and-out 900 per arm | Paired-manifest validation, Stage-0/R1--R3 typed execution boundary, gate-v1 control records, and a local-backend-only runner are present; the Panda/DROID asset adapter and CF golden-output regression remain pending |
| Matcher ablation | `calibx.benchmarks.ablations` | 300 Panda-ORB frames per matcher, submitted as 2 × 150 | Gate-v1/SAM/shared-manifest provenance and lazy external adapters are present; optional only because current formal outputs do not match the paper table |
| DROID / RoboChallenge figures | `paper/manifests` and `scripts/paper` | Qualitative selections | Do not treat as aggregate or random-sample benchmarks |

The public multi-dataset planning templates use readable camera/split aliases.
The audited CF stage-1 TOMLs instead contained one `datasets.dr` entry per
file; their evidence-backed `name` and dataset-relative path values are kept
in the public templates, while no CF absolute path is retained.

## Metric boundary

`calibx.metrics` remains the upstream-compatible success-only aggregator.
Paper reporting keeps two explicit contracts rather than silently merging
them: `summarize_refinement_add` matches d7/Table 4 continuous ADD AUC with
`error <= threshold` and all requested frames as denominator; the discrete
PCK/ADD helper uses strict integer thresholds for benchmarks that define that
official contract.  Means and medians use valid poses only.  Values are
fractions unless a table generator explicitly formats them as percentages.

## Excluded from this migration

Baxter, raw datasets, checkpoints, caches, remote outputs, and all
machine-specific launchers remain outside the public source tree.

## Qualitative-label contract

The DROID and RoboChallenge figures remain explicitly qualitative.  Figure 5
selects the best successful refinement round only after the candidate cases are
specified in a manifest.  CTRNet-X panels are named `best_cases` and
`high_error_visible_cases`; the latter are successful PnP estimates with high
ADD, not pipeline failures.
