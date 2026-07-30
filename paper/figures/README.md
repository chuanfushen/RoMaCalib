# Figure sources

This directory holds public selection metadata and path-independent figure
generators.  It never contains raw RGB frames, outputs, model checkpoints, or
server-local image paths.

`scripts/paper/build_qualitative_grid.py` accepts a separate layout JSON and
an explicit asset root.  The JSON contains only relative asset keys and one of
the documented layouts: DROID Figure 3 (4×6), DROID Figure 5 (3×6),
RoboChallenge Figure 4 (3×6), or CTRNet-X cases (3×4).
