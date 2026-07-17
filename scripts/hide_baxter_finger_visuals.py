#!/usr/bin/env python3
"""Remove only the visual elements of Baxter's moving gripper fingers."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


FINGER_LINKS = {
    "l_gripper_l_finger",
    "l_gripper_l_finger_tip",
    "l_gripper_r_finger",
    "l_gripper_r_finger_tip",
    "r_gripper_l_finger",
    "r_gripper_l_finger_tip",
    "r_gripper_r_finger",
    "r_gripper_r_finger_tip",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    args = parser.parse_args()

    tree = ET.parse(args.urdf)
    root = tree.getroot()
    changed = []
    for link in root.findall("link"):
        if link.get("name") not in FINGER_LINKS:
            continue
        visuals = list(link.findall("visual"))
        for visual in visuals:
            link.remove(visual)
        if visuals:
            changed.append(link.get("name"))

    missing = sorted(FINGER_LINKS - set(changed))
    if missing:
        raise RuntimeError(f"Expected finger visuals were not found: {missing}")
    ET.indent(tree, space="  ")
    tree.write(args.urdf, encoding="utf-8", xml_declaration=True)
    print(f"Removed moving-finger visuals from {len(changed)} links in {args.urdf}")


if __name__ == "__main__":
    main()
