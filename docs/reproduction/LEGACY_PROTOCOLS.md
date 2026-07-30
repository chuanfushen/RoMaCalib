# Historical full-run compatibility protocols

The historical DREAM-real full, CTRNet-X single-frame, and CTRNet-X batch
results predate the committed d7 gate-v1 evaluator.  Their public migration is
therefore an explicit compatibility layer, not a hidden flag alias.

## Legacy single-frame refinement

The old `refinement_iterations` behavior was:

```text
stage 0 pose
  → pose-aligned single render
  → match and PnP
  → PnP success: replace the pose unconditionally
  → PnP failure: stop the frame as failed
```

This differs from gate-v1, which evaluates candidate evidence and can reject a
candidate while retaining the parent pose.  A legacy paper regression must
keep this semantic identifier:

```text
legacy-unconditional-pnp-replace-v1
```

The formal CF stage-1 configurations used six orbit views and
`match_batch_size = 6`.  This batch size belongs to the six-view stage-1
matcher; it must not be confused with the closed-loop batch protocol below.

The public compatibility state machine reads R0 from
`<initial-results>/<frame>/frame_summary.json` and prefers the co-located
`best_pose.npz`, falling back to the summary's `pose_npz` only if the default
artifact is absent.  If `mask_input = true` and the archived Stage-1 frame
contains `input_mask.png`, it replays that exact mask on the original observed
RGB and never invokes SAM to replace the archived input.  If `mask_input =
false`, it uses the original RGB even when an archived mask exists.  A replayed
mask is copied to the new `<output>/<frame>/input_mask.png`; the archived
Stage-1 artifact is never modified.  Each R1--R3 seed is
`(base_seed + frame_index * 1009 + iteration) % (2**31 - 1)` and a runtime
hook applies it to Torch, NumPy, and OpenCV.  An unexpected exception outside
the normal PnP-failure path is caught at the outer frame boundary and produces
a frame summary with `status = "error"`; it does not fabricate an inline
`error` refinement iteration, a gate rejection, or an implicit parent
fallback.

The executable boundary is
`calibx.benchmarks.dream_real.legacy_executor` and its
`scripts/reproduce/dream_real/run_legacy_full.py` entry.  Its default mode is
plan-only.  `--execute` requires a caller-provided `initial_results_root` and
uses a lazy MuJoCo/RoMaV2 bridge only while processing a frame.  It writes new
results to its configured output root and refuses to overwrite a non-empty one
unless `resume = true`; it never modifies the Stage-1 artifact directory.
Safe resume is intentionally stricter than an output-directory check: before
the original invocation and every resume, supply the same machine-local
`--runtime-identity-file` (recording the approved code/weight/environment
identity).  The executor also hashes each selected JSON/RGB pair, consumed
Stage-1 summary/pose/mask, MuJoCo XML, and public bridge sources.  It rejects a
resume if any of those contents change; the private identity file itself is
never a public artifact.
The entry point labels a run `paper_full` only when no camera/frame restriction
is supplied and the declared scope is exactly Azure 6,394 + Kinect360 4,966 +
ORB 32,315 + RealSense 5,944 frames.  Any `--dataset`, `--frame-index`, or
`--limit` invocation must also pass `--subset` and is recorded as an explicit
subset or smoke run, never as a four-camera full result.

Archived source-run artifacts and detailed replay diagnostics are local-only
materials.  They can retain machine-specific absolute data, checkpoint, or
output paths as well as exception text and tracebacks.  Do not publish those
raw directories or full diagnostic summaries as a public result bundle; publish
only path-neutral manifests and compact, reviewed records.

For CTRNet-X Panda single-frame, use the separate
`calibx.benchmarks.ctrnetx.single_executor` boundary and
`scripts/reproduce/ctrnetx/run_single_legacy.py`.  Its portable runtime
manifest supplies a split, frame ID, explicit numeric `frame_index`, and opaque
metadata; a machine-local hook resolves assets without placing data paths in
source control.  Stage 0 is fixed at six views and `match_batch_size = 6`.
Both Stage 0 and every projected replay request fix MuJoCo `visual_geom_group`
to `2`.

```json
{
  "protocol": "ctrnetx_single_frame_runtime_frames",
  "version": 1,
  "frames": [
    {
      "split": "robot_in_view_full_body",
      "frame_id": "episode-00/frame-000017",
      "frame_index": 17,
      "metadata": {"camera": "azure"}
    }
  ]
}
```

The complete formal manifest has 1,864 / 6,264 / 9,178 records for
full-body / partial / in-and-out respectively; the one-record form above is
only a schema illustration and must be invoked with `--allow-partial`.
If its artifact has `input_mask.png`, R1--R3 must replay that exact mask; if it
does not, they use the original RGB fallback and do not invoke SAM again.  The
single-frame matcher seed for R0--R3 is
`(90 + frame_index * 1009 + iteration) % (2**31 - 1)`.
The compact boundary audit records the R0 seed and every refinement seed that
was actually attempted, including a failed PnP round whose historical detail
record did not contain a seed field.

This executor has no shared pose, retained-parent, gate, rollback, or
cross-frame recovery path.  A non-successful R1--R3 PnP marks the frame failed
and skips its later rounds.  An unexpected hook/runtime exception is recorded
at the outer frame boundary, rather than being represented as a historical
inline refinement `error` plus artificial later skips.

## Closed-loop batch

Batch evaluation estimates one shared pose per episode from top-K
correspondences of successful source frames, then evaluates that shared pose on
every requested frame in the episode.  It is a distinct protocol from
independent single-frame inference.  The public implementation must publish
the episode manifest, source-frame policy, balanced shard assignment, and any
recovery event; it may not silently skip frames.

Its stage 0 reuses the archived six-view, `match_batch_size = 6` source run.
Each subsequent replay uses one pose-aligned render per frame; that is a
per-frame rendering contract, not an evidence-backed cross-frame matcher batch
parameter.

Its source policy is intentionally asymmetric:

- i0 accepts a frame only when its archived individual Stage-1 PnP succeeded;
- i1--i3 accept raw post-geometry correspondences from the one pose-aligned
  render, even when that frame would not solve a standalone PnP;
- the shared episode pose is evaluated on every requested frame regardless of
  whether that frame entered the source pool.

If an episode solver fails after a valid parent, the state is
`retained_parent`, never a fabricated update.  If i0 has no shared parent,
later rounds are explicitly `skipped_no_parent` and still emit every frame's
evaluation record.  A backend can attach `cache_hit`,
`recompute_cached_render`, `rerender`, or `recovery_failed` audit records with
portable artifact IDs; those records are never used to hide a missing frame.

The executable public boundary is
`calibx.benchmarks.ctrnetx.batch_executor`.  It layers a runtime frame-ID and
opaque-metadata manifest on the count-only 60-episode manifest, so a local
dataset adapter can resolve assets without writing host paths into source
control.  Its hooks run Stage 1, one pose-aligned render, raw post-geometry
matching, and every-frame evaluation; the optional MuJoCo/core helpers are
lazy imports and do not change `romav2`.

For audit, the executor records both seed classes as data.  The match seed is
`(base_seed + frame_ordinal * 1009 + iteration) % (2**31 - 1)`.  The shared
PnP seed is `(base_seed + episode_ordinal * 1009) % (2**31 - 1)` at i0 and
`(base_seed + (episode_ordinal * 10 + iteration) * 1009) % (2**31 - 1)` at
i1--i3.  The frozen paper base seed is 90.  Each replay attempt stores its
match seed and each iteration record stores its PnP seed; neither is inferred
from a hidden worker counter.
