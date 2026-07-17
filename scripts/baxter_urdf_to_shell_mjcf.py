#!/usr/bin/env python3
"""Compile a Baxter URDF with material-split visual meshes into MJCF."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-urdf", type=Path, required=True)
    parser.add_argument("--output-mjcf", type=Path, required=True)
    args = parser.parse_args()

    spec = mujoco.MjSpec.from_file(args.input_urdf.as_posix())
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    model = spec.compile()
    args.output_mjcf.parent.mkdir(parents=True, exist_ok=True)
    spec.to_file(args.output_mjcf.as_posix())
    print(
        f"wrote {args.output_mjcf} "
        f"(nq={model.nq}, njnt={model.njnt}, nbody={model.nbody}, "
        f"ngeom={model.ngeom}, nmesh={model.nmesh})"
    )


if __name__ == "__main__":
    main()
