# Paper reproduction layout

The public package separates the frame-level Calib-X implementation from paper
protocols and presentation code.

```text
src/calibx/benchmarks/       dataset and protocol adapters
  dream_real/                DREAM-real full evaluation
  ctrnetx/                   Panda self-collected single and batch protocols
  ablations/                 masking and matcher ablations
src/calibx/reporting/        paper metric and provenance contracts
scripts/reproduce/           protocol planning and validation entry points
scripts/paper/               table/figure generators from compact records
config/reproduction/         shared, host-neutral TOML templates
paper/manifests/             compact public frame selections and provenance
paper/tables/, figures/      generated source and compact inputs, never raw outputs
```

For DREAM-real, `config/reproduction/dream_real/full.example.toml` is the
four-camera full-scope template.  The adjacent
`full_orb.example.toml` deliberately describes only the ORB camera subset;
it must not be used to label or report a four-camera full reproduction.

The core runner remains under `calibx.runner`.  It preserves compatibility
with the upstream success-only aggregate metrics.  Paper-specific metrics live
in `calibx.reporting.paper_metrics`: d7 refinement AUC uses its documented
continuous `<=` threshold contract and all requested frames, while official
discrete point metrics retain their own strict threshold contract.  Means and
medians use valid poses only.

The historical DREAM R0--R3, CTRNet-X independent-frame R0--R3, and CTRNet-X
episode-replay state machines are now public.  CTRNet-X single has a portable
frame manifest plus typed Stage-0/refinement hooks: it fixes Stage 0 at six
views/batch six, replays the saved Stage-0 mask if present, and never inherits
the batch protocol's shared-pose or retained-parent behavior.  DREAM-real additionally exposes
a dedicated lazy MuJoCo/RoMaV2 artifact-replay executor: it reads the prior
Stage-1 pose/mask rather than rerunning SAM or redirecting through the generic
runner.  CTRNet-X batch additionally has
a typed runtime frame manifest, Stage-1/render/match/evaluation hooks, and
optional MuJoCo/core bridges in `benchmarks/ctrnetx/batch_executor.py`; a
dataset adapter still resolves its local assets, and a CF golden-output
regression is still required.  These layers must not be silently redirected to
`calibx.runner`.  No dataset, model checkpoint, server path, full output
directory, or generated PNG is part of this tree.

The Table 4 SAM foreground ablation now likewise has a separate typed runtime
boundary in `benchmarks/ablations/sam_executor.py`.  It freezes the real d7
split between Stage 0 (six orbit renders in one batch of six) and R1--R3 (one
parent-pose render/matcher pair per round), and gate-v1 controls candidate
selection.  Its public runner only plans or validates by default; a real run
must supply a machine-local backend, and still requires CF golden-output
regression before it can support a paper claim.

See [SOURCE_MAP.md](SOURCE_MAP.md) for the evidence-to-code mapping and the
intentional separation between CF numerical protocols and SZ qualitative code.
