#!/usr/bin/env python3
"""Audit compiled MuJoCo geom groups before render matching or mesh picking."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-xml", type=Path, required=True)
    parser.add_argument("--source-visual-group", type=int, default=None)
    parser.add_argument("--expected-visual-group", type=int, default=2)
    return parser.parse_args()


def audit(
    mujoco_xml: Path,
    expected_visual_group: int,
    source_visual_group: int | None = None,
) -> dict:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(mujoco_xml))
    model = spec.compile()
    group_counts_before = Counter(int(group) for group in model.geom_group)
    remapped_geoms = 0
    if (
        source_visual_group is not None
        and source_visual_group != expected_visual_group
    ):
        source_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_group[geom_id]) == source_visual_group
        ]
        target_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_group[geom_id]) == expected_visual_group
        ]
        if target_ids:
            raise RuntimeError(
                f"Target visual group {expected_visual_group} is already non-empty"
            )
        for geom_id in source_ids:
            model.geom_group[geom_id] = expected_visual_group
        remapped_geoms = len(source_ids)
    group_counts_after = Counter(int(group) for group in model.geom_group)
    expected = []
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != expected_visual_group:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        expected.append(
            {
                "geom_id": geom_id,
                "geom_name": (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    or f"geom_{geom_id}"
                ),
                "body_id": body_id,
                "body_name": (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or f"body_{body_id}"
                ),
                "geom_type": int(model.geom_type[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
        )
    mesh_count = sum(
        item["geom_type"] == int(mujoco.mjtGeom.mjGEOM_MESH) for item in expected
    )
    return {
        "mujoco_xml": str(mujoco_xml),
        "ngeom": int(model.ngeom),
        "group_counts_before": dict(sorted(group_counts_before.items())),
        "source_visual_group": source_visual_group,
        "remapped_geoms": remapped_geoms,
        "group_counts_after": dict(sorted(group_counts_after.items())),
        "expected_visual_group": expected_visual_group,
        "expected_group_geom_count": len(expected),
        "expected_group_mesh_count": mesh_count,
        "expected_group_geoms": expected,
        "valid": bool(expected) and mesh_count > 0,
    }


def main() -> None:
    args = parse_args()
    result = audit(
        args.mujoco_xml,
        args.expected_visual_group,
        args.source_visual_group,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(
            f"Compiled model has no mesh geometry in required visual group "
            f"{args.expected_visual_group}"
        )


if __name__ == "__main__":
    main()
