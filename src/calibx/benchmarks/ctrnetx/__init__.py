"""CTRNet-X Panda reproduction protocols.

``single_executor`` is the historical independent-frame R0--R3 protocol:
six-view Stage 0 followed by unconditional-replace one-render refinements.
``batch_replay``/``batch_executor`` are a different closed-loop, shared-pose
episode protocol.  They are intentionally not imported here so planning and
``--help`` remain free of optional evaluation/runtime dependencies.
"""

__all__ = [
    "batch_executor",
    "batch_replay",
    "episode_pnp",
    "protocol",
    "single_executor",
]
