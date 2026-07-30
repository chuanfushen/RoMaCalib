# Paper source map

This map records where each public-facing protocol was migrated from.  Public
wrappers preserve the historical control flow and parameters, while private
paths, data readers, checkpoints, and outputs are isolated behind local
adapters.  It is deliberately about source semantics and scope, not a
substitute for a clean reproduction result.

| Paper scope | Evidence source | Public destination | Current state |
| --- | --- | --- | --- |
| DREAM-real full | CF formal stage-1 TOMLs and the historical pre-gate evaluator | `benchmarks/dream_real/legacy.py`, `benchmarks/dream_real/legacy_runner.py`, `benchmarks/dream_real/legacy_executor.py`, `config/reproduction/dream_real` | Four-camera plan, artifact reader, unconditional R0--R3 state machine, and lazy MuJoCo/RoMaV2 artifact-replay backend are present; CF golden-output regression remains pending |
| CTRNet-X Panda single-frame | CF formal stage-1 TOMLs and historical pre-gate refinement runner | `benchmarks/ctrnetx/single_executor.py`, `benchmarks/dream_real/legacy_runner.py`, `config/reproduction/ctrnetx/single.example.toml` | Three-split plan, portable frame manifest, typed Stage-0/refinement hooks, and legacy R0--R3 transitions are present; a local dataset hook and CF golden-output regression remain pending |
| CTRNet-X Panda closed-loop batch | CF i0/i1–i3 episode PnP/replay scripts and their shard accounting | `benchmarks/ctrnetx/episode_pnp.py`, `benchmarks/ctrnetx/batch_replay.py`, `benchmarks/ctrnetx/batch_executor.py`, batch manifests and shard planner | Public runtime frame manifest, typed Stage-1/render/raw-match/evaluation hooks, exact seed audit, and optional MuJoCo/core bridges are present; dataset adapter and CF golden-output regression pending |
| Table 4 SAM foreground ablation | CF d7 gate-v1 configuration and paired 900-frame result scope | `benchmarks/gate_v1.py`, `benchmarks/gate_rendering.py`, `benchmarks/ablations/sam.py`, `benchmarks/ablations/sam_executor.py` | Gate primitives, pose-aligned renderer, paired-manifest validation, Stage-0/R1--R3 typed executor, seed/provenance audit, and local-backend-only runner are present; asset adapter and CF golden-output regression pending |
| Table 5 feature matcher ablation | CF matcher-ablation scripts and formal-output audit | `benchmarks/ablations/matchers.py`, `benchmarks/ablations/external_matchers.py`, matcher plan | Gate-v1/SAM/two-shard shared-manifest contract and lazy external adapters are present; recovered formal values do not yet justify a paper table claim |
| DROID / RoboChallenge figures | SZ visual-selection and composition scripts | `reporting/selection.py`, `reporting/figure_grid.py`, `scripts/paper` | Path-neutral selection/building layer only; no raw outputs included |

## Source boundaries

- GitHub `main` (`48c5602`) is the clean upstream-derived baseline for this
  release worktree.
- The local `xieting` branch contains d7 gate-v1 work and other user changes;
  it is evidence, not a directory to copy wholesale.
- CF is the source of numerical experiment semantics and scope.  Its historical
  scripts include a pre-gate refinement behavior that must remain distinct from
  d7 gate-v1.
- SZ is the source of qualitative-selection/composition behavior.  It does not
  supply the numerical paper tables in this release plan.
- Baxter is intentionally excluded from this public migration stage.

## Dataset-key note

The audited CF stage-1 TOMLs held one dataset at a time under the logical key
`dr`.  The public full-scope templates use human-readable aliases such as
`azure` and `robot_in_view_partial` only to plan several audited TOMLs in one
file.  The original dataset `name` and relative data layout are preserved in
each entry; host paths and outputs are not.

## Parameter evidence

- CF formal DREAM-real and CTRNet-X single-frame stage-1/replay TOMLs use six
  orbit views and `match_batch_size = 6`.  That batch is six candidate renders
  for one observed image, never a cross-frame batch.
- The CF CTRNet-X batch worker exposes no replay `match_batch_size`: i1--i3
  render and match one pose-aligned image pair per requested frame, then solve
  one episode pose from the aggregate correspondences.
- Table 4 is d7 gate-v1, not legacy refinement.  Its final CF provenance
  records Stage-1 `--views 6 --match-batch-size 6`; its R1--R3 implementation
  has exactly one render and an internal one-pair matcher call.
