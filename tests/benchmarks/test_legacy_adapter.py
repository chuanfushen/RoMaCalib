from __future__ import annotations

from pathlib import Path

from calibx.benchmarks.dream_real.legacy import (
    LegacyEvaluatorRequest,
    request_from_paper_config,
)


def test_legacy_adapter_preserves_old_refinement_flag_and_optional_inputs() -> None:
    request = LegacyEvaluatorRequest(
        dataset_dir=Path("data/dream/panda-orb"),
        mujoco_xml=Path("assets/panda.xml"),
        prerender_dir=Path("outputs/prerender/panda-orb"),
        output_dir=Path("outputs/paper/panda-orb"),
        sam3_checkpoint=Path("weights/sam3.pt"),
        visual_geom_group=2,
        sample_count=300,
        sample_seed=90,
        views=6,
        match_batch_size=6,
        refinement_iterations=3,
        device="cuda",
        mask_prompt="robotic arm",
        mask_input=True,
        resume=False,
        save_visualizations=True,
        save_all_matches=True,
        visual_body_names=("panda",),
        match_seed=90,
        frame_indices=(1, 5),
        initial_results_dir=Path("outputs/stage1/panda-orb"),
    )
    argv = request.to_legacy_argv()
    assert "--refinement-iterations" in argv
    assert "--refinement-enabled" not in argv
    assert argv[argv.index("--refinement-iterations") + 1] == "3"
    assert "--initial-results-dir" in argv


def test_legacy_adapter_allows_a_dataset_specific_full_run_count() -> None:
    request = request_from_paper_config(
        {
            "paths": {
                "mujoco_xml": "assets/panda.xml",
                "prerender_root": "outputs/prerender",
                "output_root": "outputs/eval",
                "sam3_checkpoint": "assets/sam3.pt",
            },
            "datasets": {
                "orb": {
                    "name": "panda-orb",
                    "path": "data/orb",
                    "sample_count": 32_315,
                }
            },
            "evaluation": {
                "sample_count": 1,
                "sample_seed": 90,
                "views": 6,
                "match_batch_size": 6,
                "refinement_iterations": 3,
                "device": "cuda",
                "mask_prompt": "robotic arm",
                "mask_input": True,
                "resume": False,
                "save_visualizations": False,
                "save_all_matches": False,
            },
        },
        "orb",
    )
    assert request.sample_count == 32_315
    assert request.output_dir.name == "panda-orb_sample32315"


def test_legacy_adapter_uses_output_name_when_initial_root_has_no_override() -> None:
    config = {
        "paths": {
            "mujoco_xml": "assets/panda.xml",
            "prerender_root": "outputs/prerender",
            "output_root": "outputs/eval",
            "initial_results_root": "outputs/stage1",
            "sam3_checkpoint": "assets/sam3.pt",
        },
        "datasets": {"orb": {"name": "panda-orb", "path": "data/orb"}},
        "evaluation": {
            "sample_count": 300,
            "sample_seed": 90,
            "views": 6,
            "match_batch_size": 6,
            "refinement_iterations": 3,
            "device": "cuda",
            "mask_prompt": "robotic arm",
            "mask_input": True,
            "resume": False,
            "save_visualizations": False,
            "save_all_matches": False,
        },
    }
    request = request_from_paper_config(config, "orb")
    assert request.initial_results_dir == Path("outputs/stage1/panda-orb_sample300")
