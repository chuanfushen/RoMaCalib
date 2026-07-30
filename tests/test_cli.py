from __future__ import annotations

import pytest

from calibx.cli import build_parser, main


def test_cli_requires_a_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.command == "run"
    assert args.dry_run is True


def test_default_config_dry_run_does_not_load_models(capsys) -> None:
    assert main(["run", "--dry-run", "--datasets", "dr"]) == 0
    output = capsys.readouterr().out
    assert "calibx.runner" in output


def test_paper_protocol_template_cannot_be_run_by_the_core_runner(tmp_path) -> None:
    config = tmp_path / "paper.toml"
    config.write_text(
        """
[protocol]
id = "ctrnetx_closed_loop_batch"

[paths]
data_root = "data"
mujoco_xml = "robot.xml"
prerender_root = "renders"
output_root = "outputs"

[datasets.example]
name = "example"
path = "example"

[evaluation]
datasets = ["example"]
views = 6
match_batch_size = 1

[prerender]
workers = 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paper protocol plan"):
        main(["run", "--config", str(config), "--dry-run"])
