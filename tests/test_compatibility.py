from __future__ import annotations

import importlib.util
from pathlib import Path

from calibx.configuration import load_config as load_calibx_config
from calibx.dataset_adapters import Dream as CalibxDream
from calibx.metrics import auc_under_threshold, summarize
from calibx.pipeline import (
    _clean_record,
    _match_render_paths,
    _match_romav2_images_batched,
    _remove_frame_artifacts,
    _remove_render_artifacts,
    process_frame,
)
from calibx.rendering import load_joint_positions
from calibx.runner import _frame_index_from_json
from romav2 import RoMaV2
from romav2.benchmarks import Dream as LegacyDream
from romav2.benchmarks.utils import evaluator as legacy_evaluator
from romav2.benchmarks.utils import pose as legacy_pose
from romav2.dream_config import load_config as load_legacy_config
from romav2.romav2 import RoMaV2 as NativeRoMaV2


def test_legacy_dream_import_points_to_calibx() -> None:
    assert LegacyDream is CalibxDream


def test_legacy_configuration_import_is_preserved() -> None:
    assert load_legacy_config is load_calibx_config


def test_legacy_top_level_romav2_export_is_preserved() -> None:
    assert RoMaV2 is NativeRoMaV2


def test_legacy_evaluator_helper_exports_are_preserved() -> None:
    assert legacy_evaluator.frame_index_from_json is _frame_index_from_json
    assert legacy_evaluator.clean_record is _clean_record
    assert legacy_evaluator.remove_render_artifacts is _remove_render_artifacts
    assert legacy_evaluator.remove_frame_artifacts is _remove_frame_artifacts
    assert legacy_evaluator.match_romav2_images_batched is _match_romav2_images_batched
    assert legacy_evaluator.match_render_paths is _match_render_paths
    assert legacy_evaluator.process_frame is process_frame
    assert legacy_evaluator.auc_under_threshold is auc_under_threshold
    assert legacy_evaluator.summarize is summarize


def test_legacy_evaluator_parser_does_not_require_refactor_only_flags() -> None:
    args = legacy_evaluator.parse_args([])
    assert args.dataset_dir == legacy_evaluator.DEFAULT_DATASET_DIR
    assert args.output_dir == legacy_evaluator.DEFAULT_OUTPUT_DIR
    assert not hasattr(args, "sam3_bpe")


def test_legacy_pose_load_joint_positions_export_is_preserved() -> None:
    assert legacy_pose.load_joint_positions is load_joint_positions


def test_legacy_dream_script_exports_its_original_orchestration_helpers() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts/run_dream_eval.py"
    spec = importlib.util.spec_from_file_location("legacy_dream_runner", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.flag(True, "mask-input") == "--mask-input"
    assert module.flag(False, "mask-input") == "--no-mask-input"
    assert callable(module.run_prerender)
    assert callable(module.run_evaluation)
