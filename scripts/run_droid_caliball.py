#!/usr/bin/env python3
"""Optimize one shared DROID camera pose with multi-frame CalibAll masks.

This entry point is intentionally isolated from the main uv environment.  It
runs in the configured CalibAll environment, consumes an audited NPZ bundle,
and never reads validation or held-out masks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--history-interval", type=int, default=25)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction) > 0.5
    truth = np.asarray(target) > 0
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum() / union) if union else 0.0


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    import torch
    import torch.nn as nn

    from caliball.rendering.nvdiffrast_renderer import NVDiffrastRenderer
    from caliball.utils.transforms import se3_exp_map, se3_log_map

    bundle = np.load(args.bundle)
    vertices = torch.as_tensor(bundle["vertices"], dtype=torch.float32, device=args.device)
    faces = torch.as_tensor(bundle["faces"], dtype=torch.int32, device=args.device)
    camera_matrix = torch.as_tensor(bundle["camera_matrix"], dtype=torch.float32, device=args.device)
    targets = torch.as_tensor(bundle["target_masks"], dtype=torch.float32, device=args.device)
    initial = torch.as_tensor(bundle["initial_world_to_camera"], dtype=torch.float32, device=args.device)
    frame_indices = np.asarray(bundle["frame_indices"], dtype=np.int64)
    if vertices.ndim != 3 or targets.ndim != 3 or len(vertices) != len(targets):
        raise ValueError("Expected vertices[F,V,3] and target_masks[F,H,W]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected one shared faces[N,3] topology")
    frame_count, height, width = targets.shape

    class SharedPoseSolver(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            initial_dof = se3_log_map(
                initial[None].permute(0, 2, 1),
                eps=1e-5,
                backend="opencv",
                test_acc=False,
            )[0]
            self.dof = nn.Parameter(initial_dof, requires_grad=True)
            self.renderer = NVDiffrastRenderer([height, width], device=args.device)

        def transform(self):
            return se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]

        def render(self, frame_index: int, transform):
            return self.renderer.render_mask(
                vertices[frame_index],
                faces,
                K=camera_matrix,
                object_pose=transform,
            )

    solver = SharedPoseSolver().to(args.device)
    optimizer = torch.optim.Adam(
        solver.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2000,
        T_mult=2,
        eta_min=1e-5,
    )
    best_loss = float("inf")
    best_step = -1
    best_dof = solver.dof.detach().clone()
    history = []
    started = time.monotonic()

    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        transform = solver.transform()
        total_value = 0.0
        total_mse = 0.0
        total_iou_loss = 0.0
        for frame_ordinal in range(frame_count):
            rendered = solver.render(frame_ordinal, transform)
            target = targets[frame_ordinal]
            mse = ((rendered - target) ** 2).mean()
            intersection = (rendered * target).sum()
            union = rendered.sum() + target.sum() - intersection
            iou_loss = 1.0 - intersection / (union + 1e-6)
            loss = mse + 0.5 * iou_loss
            (loss / frame_count).backward(retain_graph=frame_ordinal + 1 < frame_count)
            total_value += float(loss.detach().cpu()) / frame_count
            total_mse += float(mse.detach().cpu()) / frame_count
            total_iou_loss += float(iou_loss.detach().cpu()) / frame_count
        torch.nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
        if total_value < best_loss:
            best_loss = total_value
            best_step = step
            best_dof = solver.dof.detach().clone()
        if step % args.history_interval == 0 or step == args.max_steps - 1:
            record = {
                "step": step,
                "loss": total_value,
                "mse": total_mse,
                "soft_iou_loss": total_iou_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            print(
                f"step={step}/{args.max_steps} loss={total_value:.6f} "
                f"mse={total_mse:.6f} iou_loss={total_iou_loss:.6f} best={best_loss:.6f}",
                flush=True,
            )
        optimizer.step()
        scheduler.step(step)

    with torch.no_grad():
        solver.dof.copy_(best_dof)
        refined = solver.transform()
        initial_ious = []
        refined_ious = []
        initial_dof = se3_log_map(
            initial[None].permute(0, 2, 1), eps=1e-5, backend="opencv", test_acc=False
        )[0]
        for frame_ordinal in range(frame_count):
            solver.dof.copy_(initial_dof)
            initial_rendered = solver.render(frame_ordinal, solver.transform())
            initial_ious.append(mask_iou(initial_rendered.cpu().numpy(), targets[frame_ordinal].cpu().numpy()))
            solver.dof.copy_(best_dof)
            refined_rendered = solver.render(frame_ordinal, solver.transform())
            refined_ious.append(mask_iou(refined_rendered.cpu().numpy(), targets[frame_ordinal].cpu().numpy()))
        solver.dof.copy_(best_dof)
        refined = solver.transform().detach().cpu().numpy().astype(np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "fit_result.npz",
        initial_world_to_camera=np.asarray(bundle["initial_world_to_camera"], dtype=np.float64),
        refined_world_to_camera=refined,
        best_dof=best_dof.detach().cpu().numpy(),
        frame_indices=frame_indices,
        initial_fit_ious=np.asarray(initial_ious, dtype=np.float64),
        refined_fit_ious=np.asarray(refined_ious, dtype=np.float64),
    )
    write_json(
        args.output_dir / "fit_summary.json",
        {
            "implementation": "droid_multiframe_caliball_integration_v0",
            "bundle": str(args.bundle),
            "device": args.device,
            "frame_count": int(frame_count),
            "frame_indices": frame_indices.tolist(),
            "fit_size": [int(width), int(height)],
            "max_steps": args.max_steps,
            "optimizer": {
                "name": "Adam",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "gradient_clip_norm": 1.0,
                "scheduler": "CosineAnnealingWarmRestarts(T_0=2000,T_mult=2,eta_min=1e-5)",
            },
            "objective": "frame_mean(mean_squared_mask_error + 0.5 * soft_iou_loss)",
            "selection_policy": "best_fit_loss_checkpoint",
            "selected_step": best_step,
            "best_loss": best_loss,
            "initial_fit_iou_macro": float(np.mean(initial_ious)),
            "refined_fit_iou_macro": float(np.mean(refined_ious)),
            "history": history,
            "elapsed_seconds": time.monotonic() - started,
            "refined_world_to_camera": refined.tolist(),
        },
    )


if __name__ == "__main__":
    main()
