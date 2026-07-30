# Reproduction configuration

Each public protocol has a host-neutral `.example.toml` below this directory.
Machine-specific data roots, checkpoints, prerender directories, and output
roots must be provided through an ignored `config/reproduction/**/*.local.toml`
config or documented environment variables.  Public templates never contain
server absolute paths or `resume = true` defaults.  The four-camera
DREAM-real full template is `dream_real/full.example.toml`; by contrast,
`dream_real/full_orb.example.toml` is an ORB camera-subset template and must
not be labelled or reported as a four-camera full result.  The DREAM-real
templates and CTRNet-X single-frame templates preserve their historical legacy
semantic identifier.  The DREAM-real executor requires a host-local
`initial_results_root` with the matching Stage-1 artifacts; it does not
recreate Stage 1 or SAM.  CTRNet-X single uses a separate portable runtime
frame manifest and typed local hooks: R0 is six-view/batch-six, R1--R3 replay
the existing Stage-0 mask when present, and no batch/gate recovery logic is
allowed.  CTRNet-X batch is a
separate closed-loop protocol with six stage-1 views and one replay render per
iteration, rather than a cross-frame matcher batch.  SAM uses the separate d7
gate-v1 primitive contract: Stage 1 uses six views/batch 6 and each refinement
round has one pose-aligned render/one-pair matcher call.  Its public typed
execution boundary records Stage-0/R1--R3 control and evidence data, but a
machine-local backend still resolves the opaque Panda/DROID frame IDs and a CF
golden-output regression remains required.  The Table 5 matcher template
separately fixes gate-v1, the SAM input mask, and a 300-frame parent manifest
split into two 150-frame shards; it records only the generic provenance shape,
not historical server hashes or paper values.
