#!/usr/bin/env python3
"""Convert the CtRNet Baxter real dataset into the local DREAM-style layout.

The source dataset contains 100 flattened PNG images, 100 rows of left-arm
joint angles, and one 2D/3D end-effector annotation for each of 20 poses.
The output is intentionally created in a new directory and is never
overwritten by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


OFFICIAL_INTRINSICS = {
    "fx": 960.41357421875,
    "fy": 960.22314453125,
    "cx": 1021.7171020507812,
    "cy": 776.2381591796875,
}
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
LEFT_JOINT_NAMES = (
    "left_s0",
    "left_s1",
    "left_e0",
    "left_e1",
    "left_w0",
    "left_w1",
    "left_w2",
)
OFFICIAL_REPOSITORY_COMMIT = "05156c5ed5ac4bd67e96b51dc72238ff2743c388"


class RestrictedUnpickler(pickle.Unpickler):
    """Load the primitive-only legacy annotation without importing globals."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/baxter_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/baxter_real_dream_ctrnet_ee"))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = RestrictedUnpickler(stream).load()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    image_dir = source_dir / "images"
    joint_path = source_dir / "gt_joint.npy"
    annotation_path = source_dir / "baxter-real-dataset" / "ground_truth_data"
    images = [image_dir / f"{index}.png" for index in range(100)]
    missing = [path for path in (*images, joint_path, annotation_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing[:5]}")

    joints = np.load(joint_path, allow_pickle=False)
    ground_truth = load_ground_truth(annotation_path)
    if joints.shape != (100, 7):
        raise ValueError(f"Expected gt_joint.npy shape (100, 7), got {joints.shape}")
    if set(ground_truth) != {f"pose_{index}" for index in range(20)}:
        raise ValueError("ground_truth_data must contain exactly pose_0 through pose_19")

    for pose_index in range(20):
        pose_joints = np.asarray(
            ground_truth[f"pose_{pose_index}"]["joints"], dtype=np.float64
        )
        if not np.allclose(joints[pose_index * 5 : (pose_index + 1) * 5], pose_joints):
            raise ValueError(
                f"gt_joint.npy rows for pose_{pose_index} do not match ground_truth_data"
            )

    output_dir.mkdir(parents=True)
    frame_records = []
    for frame_index, source_image in enumerate(images):
        pose_index = frame_index // 5
        pose_gt = ground_truth[f"pose_{pose_index}"]
        frame_stem = f"{frame_index:06d}"
        output_image = output_dir / f"{frame_stem}.rgb.jpg"
        output_json = output_dir / f"{frame_stem}.json"

        with Image.open(source_image) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(
                    f"{source_image} has size {image.size}, expected "
                    f"{(IMAGE_WIDTH, IMAGE_HEIGHT)}"
                )
            image.convert("RGB").save(
                output_image,
                format="JPEG",
                quality=args.jpeg_quality,
                subsampling=0,
                optimize=True,
            )

        joint_entries = [
            {"name": name, "position": float(position)}
            for name, position in zip(LEFT_JOINT_NAMES, joints[frame_index])
        ]
        frame_payload = {
            "objects": [
                {
                    "class": "Baxter",
                    "visibility": 1,
                    "keypoints": [
                        {
                            "name": "ctrnet_baxter_ee",
                            "location": [float(value) for value in pose_gt["ee_3d"]],
                            "projected_location": [
                                float(value) for value in pose_gt["ee_2d"]
                            ],
                        }
                    ],
                }
            ],
            "sim_state": {"joints": joint_entries},
            "source_metadata": {
                "source_image": str(source_image.relative_to(source_dir)),
                "source_frame_index": frame_index,
                "pose_index": pose_index,
                "images_per_pose": 5,
                "annotated_robot_part": "Baxter left arm",
            },
        }
        write_json(output_json, frame_payload)
        frame_records.append(
            {
                "frame": frame_stem,
                "pose_index": pose_index,
                "source_image": str(source_image.relative_to(source_dir)),
                "source_sha256": sha256(source_image),
                "output_image_sha256": sha256(output_image),
            }
        )

    camera_payload = {
        "camera_settings": [
            {
                "id": "ctrnet_baxter_azure_kinect",
                "name": "CtRNet Baxter Azure Kinect RGB camera",
                "intrinsic_settings": {
                    **OFFICIAL_INTRINSICS,
                    "s": 0.0,
                    "resolution": {
                        "width": IMAGE_WIDTH,
                        "height": IMAGE_HEIGHT,
                    },
                },
                "captured_image_size": {
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                },
            }
        ]
    }
    write_json(output_dir / "_camera_settings.json", camera_payload)
    write_json(
        output_dir / "_object_settings.json",
        {
            "exported_object_classes": ["Baxter"],
            "exported_objects": [
                {
                    "class": "Baxter",
                    "segmentation_class_id": 1,
                    "segmentation_instance_id": 0,
                }
            ],
        },
    )

    manifest = {
        "format": "RoMaCalib DREAM-compatible Baxter real dataset",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "frame_count": 100,
        "pose_count": 20,
        "images_per_pose": 5,
        "image_size": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "output_image_format": {
            "format": "JPEG",
            "quality": args.jpeg_quality,
            "subsampling": 0,
        },
        "camera_intrinsics": {
            **OFFICIAL_INTRINSICS,
            "source": "Official CtRNet train.py and evaluate_baxter_dataset.ipynb",
            "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
            "distortion_coefficients": None,
        },
        "joint_annotation": {
            "names_in_source_order": list(LEFT_JOINT_NAMES),
            "annotated_part": "left arm only",
            "right_arm_joint_state_available": False,
            "pipeline_behavior_for_unlisted_joints": "MuJoCo model default qpos",
            "warning": (
                "The source dataset does not provide right-arm joint angles. "
                "Unlisted joints must not be interpreted as measured zero angles."
            ),
        },
        "evaluation_annotation": {
            "keypoint_count_per_frame": 1,
            "keypoint_name": "ctrnet_baxter_ee",
            "ground_truth": "2D and 3D end-effector position in the camera frame",
            "metric_scope": (
                "Single end-effector 3D distance and pixel/PCK error, not full-robot "
                "multi-keypoint ADD."
            ),
            "prediction_definition": (
                "Official CtRNet Baxter DH chain evaluated from the fixed "
                "left_arm_mount frame with T_7_ee translation [0, 0, 0.3683] m."
            ),
            "not_a_mesh_body": True,
        },
        "fitted_intrinsics": {
            "used": False,
            "note": (
                "No fitted intrinsics were written or retained. Conversion uses "
                "the official CtRNet values."
            ),
        },
        "frames": frame_records,
    }
    write_json(output_dir / "conversion_manifest.json", manifest)

    print(f"Converted {len(frame_records)} frames to {output_dir}")
    print(f"Camera intrinsics: {OFFICIAL_INTRINSICS}")
    print("Annotated joints: " + ", ".join(LEFT_JOINT_NAMES))
    print("Evaluation keypoint: ctrnet_baxter_ee (official single end-effector point)")


if __name__ == "__main__":
    main()
