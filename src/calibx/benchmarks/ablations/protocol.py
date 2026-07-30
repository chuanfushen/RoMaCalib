"""Invariants shared by the paper's paired ablation studies."""

from __future__ import annotations

from dataclasses import dataclass

from ..common import FrameManifest, assert_paired_manifests


@dataclass(frozen=True)
class PairedAblation:
    """A treatment/control comparison that must share all requested frames."""

    name: str
    control_manifest: FrameManifest
    treatment_manifest: FrameManifest
    control_label: str
    treatment_label: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ablation name must not be empty")
        if self.control_label == self.treatment_label:
            raise ValueError("ablation labels must differ")
        assert_paired_manifests(self.control_manifest, self.treatment_manifest)


__all__ = ["PairedAblation"]
