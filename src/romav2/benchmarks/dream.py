from __future__ import annotations

import json
from pathlib import Path


class Dream:
    """DREAM robot pose benchmark backed by batched render matching and PnP."""

    def __init__(
        self,
        data_root: str | Path,
        prerender_root: str | Path,
        mujoco_xml: str | Path,
        output_dir: str | Path,
        *,
        sample_count: int | None = 300,
        sample_seed: int = 90,
        views: int = 6,
        match_batch_size: int = 6,
        visual_geom_group: int = 2,
        device: str = "cuda",
        mask_input: bool = True,
        mask_prompt: str = "robotic arm",
        sam3_checkpoint: str | Path | None = None,
        resume: bool = False,
        save_visualizations: bool = True,
        save_all_matches: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.prerender_root = Path(prerender_root)
        self.mujoco_xml = Path(mujoco_xml)
        self.output_dir = Path(output_dir)
        self.sample_count = sample_count
        self.sample_seed = sample_seed
        self.views = views
        self.match_batch_size = match_batch_size
        self.visual_geom_group = visual_geom_group
        self.device = device
        self.mask_input = mask_input
        self.mask_prompt = mask_prompt
        self.sam3_checkpoint = None if sam3_checkpoint is None else Path(sam3_checkpoint)
        self.resume = resume
        self.save_visualizations = save_visualizations
        self.save_all_matches = save_all_matches

    def _args(self):
        from .utils.evaluator import parse_args

        argv = [
            "--dataset-dir", str(self.data_root),
            "--prerender-dir", str(self.prerender_root),
            "--mujoco-xml", str(self.mujoco_xml),
            "--visual-geom-group", str(self.visual_geom_group),
            "--output-dir", str(self.output_dir),
            "--sample-seed", str(self.sample_seed),
            "--views", str(self.views),
            "--match-batch-size", str(self.match_batch_size),
            "--device", self.device,
            "--mask-prompt", self.mask_prompt,
            "--mask-input" if self.mask_input else "--no-mask-input",
            "--resume" if self.resume else "--no-resume",
            "--save-visualizations" if self.save_visualizations else "--no-save-visualizations",
            "--save-all-matches" if self.save_all_matches else "--no-save-all-matches",
        ]
        if self.sample_count is not None:
            argv.extend(("--sample-count", str(self.sample_count)))
        if self.sam3_checkpoint is not None:
            argv.extend(("--sam3-checkpoint", str(self.sam3_checkpoint)))
        return parse_args(argv)

    def benchmark(self, model, model_name=None) -> dict:
        from .utils.evaluator import process
        from .utils.geometry import ImcuiMatcher

        args = self._args()
        output_dir = process(args, matcher_api=ImcuiMatcher(args, matcher=model))
        payload = json.loads((output_dir / "summary.json").read_text())
        result = dict(payload["summary"])
        if model_name is not None:
            result["model_name"] = model_name
        return result
