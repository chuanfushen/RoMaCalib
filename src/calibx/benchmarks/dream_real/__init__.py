"""DREAM-real reproduction protocol and artifact-replay executor.

Dataset files, server paths, checkpoints, and historical outputs remain
machine-local.  The public executor consumes a caller-provided Stage-1 result
directory rather than rebuilding that stage through the generic runner.
"""
