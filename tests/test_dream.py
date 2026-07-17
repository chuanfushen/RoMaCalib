from __future__ import annotations

import logging
from functools import lru_cache

from romav2 import RoMaV2
from romav2.benchmarks import Dream
from romav2.dream_config import load_config, project_path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> RoMaV2:
    model = RoMaV2()
    model.apply_setting("precise")
    return model


def benchmark_dataset(dataset_key: str) -> dict:
    config, _ = load_config()
    dataset = config["datasets"][dataset_key]
    evaluation = config["evaluation"]
    refinement = config.get("refinement", {})
    camera = config.get("camera", {})
    benchmark = Dream(
        data_root=project_path(dataset["path"]),
        prerender_root=project_path(config["paths"]["prerender_root"]) / dataset["name"],
        mujoco_xml=project_path(config["paths"]["mujoco_xml"]),
        output_dir=(
            project_path(config["paths"]["output_root"])
            / f"{dataset['name']}_sample{evaluation['sample_count']}"
        ),
        sample_count=evaluation["sample_count"],
        sample_seed=evaluation["sample_seed"],
        views=evaluation["views"],
        width=refinement.get("process_width"),
        height=refinement.get("process_height"),
        match_batch_size=evaluation["match_batch_size"],
        visual_geom_group=int(config.get("robot", {}).get("visual_geom_group", 2)),
        source_visual_geom_group=config.get("robot", {}).get(
            "source_visual_geom_group"
        ),
        visual_body_names=config.get("robot", {}).get("visual_body_names"),
        device=evaluation["device"],
        mask_input=evaluation["mask_input"],
        mask_prompt=evaluation["mask_prompt"],
        sam3_checkpoint=project_path(config["paths"]["sam3_checkpoint"]),
        resume=evaluation["resume"],
        save_visualizations=evaluation["save_visualizations"],
        save_all_matches=evaluation["save_all_matches"],
        refinement_enabled=bool(refinement.get("enabled", False)),
        refine_iterations=int(refinement.get("iterations", 0)),
        save_refinement_artifacts=bool(refinement.get("save_artifacts", True)),
        distortion_state=str(camera.get("distortion_state", "unknown")),
        distortion_coefficients=camera.get("distortion_coefficients"),
    )
    result = benchmark.benchmark(get_model(), model_name="RoMaV2")
    assert result["n_frames"] == evaluation["sample_count"]
    assert 0.0 <= result["success_rate"] <= 100.0
    logger.info("DREAM %s results: %s", dataset_key, result)
    return result


def test_dream_real():
    benchmark_dataset("real")


def test_dream_dr():
    benchmark_dataset("dr")


def test_dream_photo():
    benchmark_dataset("photo")


if __name__ == "__main__":
    for key in ("real", "dr", "photo"):
        benchmark_dataset(key)
