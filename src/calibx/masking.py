"""SAM3-based mask extraction for robot segmentation."""

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


class Sam3Extractor:
    def __init__(
        self,
        bpe_path: str | None = None,
        ckpt_path: str | None = None,
        device=None,
    ):
        self.bpe_path = bpe_path
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam3_image_model(
            bpe_path=self.bpe_path,
            checkpoint_path=self.ckpt_path,
            device=self.device,
        )
        self.processor = Sam3Processor(self.model, device=self.device)

    def extract_masks(self, img_pil, prompt="robotic arm"):
        if isinstance(img_pil, str):
            img_pil = Image.open(img_pil)
        elif isinstance(img_pil, np.ndarray):
            img_pil = Image.fromarray(img_pil)
        else:
            assert isinstance(img_pil, Image.Image)
        with torch.autocast(
            self.device,
            dtype=torch.bfloat16,
            enabled=self.device == "cuda",
        ):
            inference_state = self.processor.set_image(img_pil)
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )
        masks = output["masks"]
        if len(masks) > 0:
            best_mask = masks[0]
            return best_mask
        return None
