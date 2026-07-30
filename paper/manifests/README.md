# Selection manifests

Paper figure manifests record dataset-relative result identifiers, the explicit
selection rule, and any reused pose.  They do not record raw image paths,
server directories, checkpoints, or rendered outputs.

Figures 3--5 are qualitative artifacts.  Their final manifests will preserve
that classification and cannot be used as aggregate benchmark inputs.  In
particular, Figure 3 selects six DROID cases by `Ours IoU >= 0.80`, IoU gain,
and an at-most-one-case-per-episode constraint; it is not a top-IoU table.
