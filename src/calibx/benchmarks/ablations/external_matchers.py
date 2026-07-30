"""Lazy optional adapters for the audited Table 5 matcher candidates.

The historical experiment used local installations of RoMa v1, LightGlue, and
MASt3R beside Calib-X.  Those projects, their licenses, and their weights are
not bundled here.  This module keeps the small image-to-correspondence adapter
at a public boundary: importing it never imports an optional project, and a
caller must provide any required local assets explicitly.

It is intentionally not a paper-value reproducer.  Use it only together with
the Table 5 protocol and provenance contracts in :mod:`.matchers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np


ExternalMatcherName = Literal["romav1", "lightglue", "mast3r"]
EXTERNAL_MATCHER_NAMES: tuple[ExternalMatcherName, ...] = (
    "romav1",
    "lightglue",
    "mast3r",
)
_CLI_NAMES: dict[ExternalMatcherName, str] = {
    "romav1": "RoMaV1",
    "lightglue": "LightGlue",
    "mast3r": "MASt3R",
}


class MatchPrediction(TypedDict):
    """Matcher output accepted by the shared render-matching pipeline."""

    mkeypoints0_orig: np.ndarray
    mkeypoints1_orig: np.ndarray
    mconf: np.ndarray


class ExternalImageMatcher(Protocol):
    """Callable adapter protocol with RGB uint8 image inputs."""

    def __call__(self, image0: np.ndarray, image1: np.ndarray) -> MatchPrediction:
        """Return real-image points, render-image points, and confidences."""


@dataclass(frozen=True)
class ExternalMatcherSettings:
    """Explicit, path-neutral settings for one optional matcher adapter.

    No checkpoint path has a default.  RoMa v1 and MASt3R require the relevant
    caller-owned local checkpoint.  LightGlue remains opt-in because its
    third-party provider controls how pretrained weights are resolved.
    """

    matcher: ExternalMatcherName
    device: str = "cuda"
    max_keypoints: int = 2048
    detect_threshold: float = 0.0005
    match_threshold: float = 0.1
    lightglue_resize: int = 0
    allow_lightglue_provider_download: bool = False
    roma_v1_custom_corr: bool = False
    roma_v1_checkpoint: Path | None = None
    dinov2_checkpoint: Path | None = None
    mast3r_checkpoint: Path | None = None
    mast3r_image_size: int = 512
    mast3r_subsample: int = 8

    def validate(self) -> None:
        if self.matcher not in EXTERNAL_MATCHER_NAMES:
            raise ValueError(f"unsupported external matcher: {self.matcher}")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.max_keypoints < 1:
            raise ValueError("max_keypoints must be positive")
        if self.detect_threshold < 0 or self.match_threshold < 0:
            raise ValueError("matcher thresholds must be non-negative")
        if self.lightglue_resize < 0:
            raise ValueError("lightglue_resize must be zero or positive")
        if self.mast3r_image_size < 16 or self.mast3r_subsample < 1:
            raise ValueError("MASt3R image_size and subsample must be positive")
        if self.matcher == "romav1" and (
            self.roma_v1_checkpoint is None or self.dinov2_checkpoint is None
        ):
            raise ValueError("romav1 requires explicit RoMa v1 and DINOv2 checkpoints")
        if self.matcher == "mast3r" and self.mast3r_checkpoint is None:
            raise ValueError("mast3r requires an explicit MASt3R checkpoint")
        if self.matcher == "lightglue" and not self.allow_lightglue_provider_download:
            raise ValueError(
                "lightglue requires explicit opt-in after its provider weights are "
                "installed or otherwise made available"
            )


def matcher_cli_name(matcher: ExternalMatcherName) -> str:
    """Return the audited CLI spelling without importing an optional package."""

    try:
        return _CLI_NAMES[matcher]
    except KeyError as error:
        raise ValueError(f"unsupported external matcher: {matcher}") from error


def empty_prediction() -> MatchPrediction:
    """Return the no-correspondence representation used by the shared pipeline."""

    return {
        "mkeypoints0_orig": np.empty((0, 2), dtype=np.float32),
        "mkeypoints1_orig": np.empty((0, 2), dtype=np.float32),
        "mconf": np.empty((0,), dtype=np.float32),
    }


def make_prediction(
    points0: np.ndarray,
    points1: np.ndarray,
    confidence: np.ndarray,
) -> MatchPrediction:
    """Normalize the audited adapters' output to a single safe contract."""

    normalized0 = np.asarray(points0, dtype=np.float32)
    normalized1 = np.asarray(points1, dtype=np.float32)
    normalized_confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if normalized0.ndim != 2 or normalized0.shape[1:] != (2,):
        raise ValueError("mkeypoints0_orig must have shape [N, 2]")
    if normalized1.ndim != 2 or normalized1.shape[1:] != (2,):
        raise ValueError("mkeypoints1_orig must have shape [N, 2]")
    if not (
        len(normalized0) == len(normalized1) == len(normalized_confidence)
    ):
        raise ValueError("matcher points and confidences must have the same length")
    return {
        "mkeypoints0_orig": normalized0,
        "mkeypoints1_orig": normalized1,
        "mconf": normalized_confidence,
    }


def restore_mast3r_coordinates(
    points: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    crop_left: float,
    crop_top: float,
) -> np.ndarray:
    """Map MASt3R crop coordinates back to the original RGB image."""

    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("MASt3R image scales must be positive")
    restored = np.asarray(points, dtype=np.float32).copy()
    if restored.ndim != 2 or restored.shape[1:] != (2,):
        raise ValueError("MASt3R points must have shape [N, 2]")
    restored[:, 0] = (restored[:, 0] + crop_left) / scale_x
    restored[:, 1] = (restored[:, 1] + crop_top) / scale_y
    return restored


def _existing_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} must be supplied explicitly")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _optional_dependency_error(name: str, error: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"{name} is an optional external matcher dependency. Install it outside "
        "the Calib-X package and provide its locally licensed assets explicitly."
    )


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("external matcher inputs must be RGB images with shape [H, W, 3]")
    return rgb


class _RoMaV1Matcher:
    def __init__(self, settings: ExternalMatcherSettings) -> None:
        try:
            import torch
            from romatch import roma_outdoor
        except ModuleNotFoundError as error:
            raise _optional_dependency_error("RoMa v1 / romatch", error) from error

        roma_v1_checkpoint = _existing_file(
            settings.roma_v1_checkpoint, "RoMa v1 checkpoint"
        )
        dinov2_checkpoint = _existing_file(settings.dinov2_checkpoint, "DINOv2 checkpoint")
        weights = torch.load(roma_v1_checkpoint, map_location="cpu", weights_only=True)
        dinov2_weights = torch.load(
            dinov2_checkpoint, map_location="cpu", weights_only=True
        )
        torch.set_float32_matmul_precision("highest")
        self._torch = torch
        self._max_keypoints = settings.max_keypoints
        self._matcher = roma_outdoor(
            device=settings.device,
            weights=weights,
            dinov2_weights=dinov2_weights,
            use_custom_corr=settings.roma_v1_custom_corr,
        ).eval()

    def __call__(self, image0: np.ndarray, image1: np.ndarray) -> MatchPrediction:
        from PIL import Image

        observed = _as_rgb_image(image0)
        rendered = _as_rgb_image(image1)
        with self._torch.inference_mode():
            pil0 = Image.fromarray(observed, mode="RGB")
            pil1 = Image.fromarray(rendered, mode="RGB")
            warp, certainty = self._matcher.match(pil0, pil1)
            matches, confidence = self._matcher.sample(
                warp, certainty, num=self._max_keypoints
            )
            h0, w0 = observed.shape[:2]
            h1, w1 = rendered.shape[:2]
            points0, points1 = self._matcher.to_pixel_coordinates(
                matches, h0, w0, h1, w1
            )
        return make_prediction(
            points0.detach().cpu().numpy(),
            points1.detach().cpu().numpy(),
            confidence.detach().cpu().numpy(),
        )


class _LightGlueMatcher:
    def __init__(self, settings: ExternalMatcherSettings) -> None:
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
        except ModuleNotFoundError as error:
            raise _optional_dependency_error("LightGlue", error) from error

        self._torch = torch
        self._device = settings.device
        self._max_keypoints = settings.max_keypoints
        self._resize = settings.lightglue_resize
        self._extractor = (
            SuperPoint(
                max_num_keypoints=settings.max_keypoints,
                detection_threshold=settings.detect_threshold,
            )
            .eval()
            .to(settings.device)
        )
        self._matcher = (
            LightGlue(
                features="superpoint",
                filter_threshold=settings.match_threshold,
            )
            .eval()
            .to(settings.device)
        )

    def __call__(self, image0: np.ndarray, image1: np.ndarray) -> MatchPrediction:
        observed = _as_rgb_image(image0)
        rendered = _as_rgb_image(image1)
        with self._torch.inference_mode():
            tensor0 = (
                self._torch.from_numpy(np.ascontiguousarray(observed).copy())
                .permute(2, 0, 1)
                .to(self._device, dtype=self._torch.float32)
                / 255.0
            )
            tensor1 = (
                self._torch.from_numpy(np.ascontiguousarray(rendered).copy())
                .permute(2, 0, 1)
                .to(self._device, dtype=self._torch.float32)
                / 255.0
            )
            resize = None if self._resize <= 0 else self._resize
            features0 = self._extractor.extract(tensor0, resize=resize)
            features1 = self._extractor.extract(tensor1, resize=resize)
            output = self._matcher({"image0": features0, "image1": features1})
            matches = output["matches"][0]
            if matches.numel() == 0:
                return empty_prediction()
            confidence = output["scores"][0]
            points0 = features0["keypoints"][0, matches[:, 0]]
            points1 = features1["keypoints"][0, matches[:, 1]]
            if confidence.numel() > self._max_keypoints:
                keep = self._torch.topk(confidence, self._max_keypoints, sorted=True).indices
                points0, points1, confidence = (
                    points0[keep],
                    points1[keep],
                    confidence[keep],
                )
        return make_prediction(
            points0.detach().cpu().numpy(),
            points1.detach().cpu().numpy(),
            confidence.detach().cpu().numpy(),
        )


class _MASt3RMatcher:
    def __init__(self, settings: ExternalMatcherSettings) -> None:
        try:
            import torch
            from mast3r.model import AsymmetricMASt3R
        except ModuleNotFoundError as error:
            raise _optional_dependency_error("MASt3R", error) from error

        checkpoint = _existing_file(settings.mast3r_checkpoint, "MASt3R checkpoint")
        self._torch = torch
        self._device = settings.device
        self._max_keypoints = settings.max_keypoints
        self._image_size = settings.mast3r_image_size
        self._subsample = settings.mast3r_subsample
        self._matcher = AsymmetricMASt3R.from_pretrained(str(checkpoint)).eval().to(
            settings.device
        )

    @staticmethod
    def _prepare_image(
        image: np.ndarray, image_size: int, index: int
    ) -> tuple[dict, dict[str, float]]:
        from PIL import Image
        from dust3r.utils.image import ImgNorm, _resize_pil_image

        pil = Image.fromarray(_as_rgb_image(image), mode="RGB")
        original_width, original_height = pil.size
        resized = _resize_pil_image(pil, image_size)
        resized_width, resized_height = resized.size
        center_x, center_y = resized_width // 2, resized_height // 2
        half_width = ((2 * center_x) // 16) * 16 / 2
        half_height = ((2 * center_y) // 16) * 16 / 2
        if resized_width == resized_height:
            half_height = 3 * half_width / 4
        left = center_x - half_width
        top = center_y - half_height
        cropped = resized.crop((left, top, center_x + half_width, center_y + half_height))
        processed_width, processed_height = cropped.size
        view = {
            "img": ImgNorm(cropped)[None],
            "true_shape": np.int32([[processed_height, processed_width]]),
            "idx": index,
            "instance": str(index),
        }
        transform = {
            "scale_x": resized_width / original_width,
            "scale_y": resized_height / original_height,
            "crop_left": float(left),
            "crop_top": float(top),
        }
        return view, transform

    def __call__(self, image0: np.ndarray, image1: np.ndarray) -> MatchPrediction:
        try:
            from dust3r.inference import inference
            from mast3r.fast_nn import fast_reciprocal_NNs
        except ModuleNotFoundError as error:
            raise _optional_dependency_error("MASt3R / DUSt3R", error) from error

        with self._torch.inference_mode():
            view0, transform0 = self._prepare_image(image0, self._image_size, 0)
            view1, transform1 = self._prepare_image(image1, self._image_size, 1)
            output = inference(
                [(view0, view1)],
                self._matcher,
                self._device,
                batch_size=1,
                verbose=False,
            )
            descriptor0 = output["pred1"]["desc"].squeeze(0).detach()
            descriptor1 = output["pred2"]["desc"].squeeze(0).detach()
            points0, points1 = fast_reciprocal_NNs(
                descriptor0,
                descriptor1,
                subsample_or_initxy1=self._subsample,
                device=self._device,
                dist="dot",
                block_size=2**13,
            )
            if len(points0) == 0:
                return empty_prediction()
            height0, width0 = output["view1"]["true_shape"][0]
            height1, width1 = output["view2"]["true_shape"][0]
            valid0 = (
                (points0[:, 0] >= 3)
                & (points0[:, 0] < int(width0) - 3)
                & (points0[:, 1] >= 3)
                & (points0[:, 1] < int(height0) - 3)
            )
            valid1 = (
                (points1[:, 0] >= 3)
                & (points1[:, 0] < int(width1) - 3)
                & (points1[:, 1] >= 3)
                & (points1[:, 1] < int(height1) - 3)
            )
            points0, points1 = points0[valid0 & valid1], points1[valid0 & valid1]
            if len(points0) == 0:
                return empty_prediction()
            confidence0 = output["pred1"]["desc_conf"].squeeze(0)
            confidence1 = output["pred2"]["desc_conf"].squeeze(0)
            confidence = self._torch.minimum(
                confidence0[points0[:, 1], points0[:, 0]],
                confidence1[points1[:, 1], points1[:, 0]],
            ).detach().cpu().numpy()
            if len(confidence) > self._max_keypoints:
                keep = np.argpartition(confidence, -self._max_keypoints)[
                    -self._max_keypoints :
                ]
                keep = keep[np.argsort(confidence[keep])[::-1]]
                points0, points1, confidence = (
                    points0[keep],
                    points1[keep],
                    confidence[keep],
                )
        return make_prediction(
            restore_mast3r_coordinates(points0, **transform0),
            restore_mast3r_coordinates(points1, **transform1),
            confidence,
        )


def create_external_matcher(settings: ExternalMatcherSettings) -> ExternalImageMatcher:
    """Construct one optional adapter only when the caller explicitly runs it."""

    settings.validate()
    if settings.matcher == "romav1":
        return _RoMaV1Matcher(settings)
    if settings.matcher == "lightglue":
        return _LightGlueMatcher(settings)
    if settings.matcher == "mast3r":
        return _MASt3RMatcher(settings)
    raise AssertionError("validated external matcher must be supported")


__all__ = [
    "EXTERNAL_MATCHER_NAMES",
    "ExternalImageMatcher",
    "ExternalMatcherName",
    "ExternalMatcherSettings",
    "MatchPrediction",
    "create_external_matcher",
    "empty_prediction",
    "make_prediction",
    "matcher_cli_name",
    "restore_mast3r_coordinates",
]
