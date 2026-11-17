"""Image transforms. Product crops have very different aspect ratios, so the default
pads to a square before resizing instead of stretching or center-cropping."""
from __future__ import annotations

from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    def __init__(self, fill=(124, 116, 104)):
        self.fill = tuple(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        if width == height:
            return image
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


def _resize(data_cfg: dict) -> list:
    size = int(data_cfg["image_size"])
    mode = data_cfg.get("resize", "pad")
    if mode == "pad":
        return [PadToSquare(), T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
    if mode == "stretch":
        return [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
    raise ValueError(f"data.resize must be pad or stretch, got {mode!r}")


def eval_transform(data_cfg: dict) -> T.Compose:
    return T.Compose([*_resize(data_cfg), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(data_cfg: dict) -> T.Compose:
    size = int(data_cfg["image_size"])
    aug = data_cfg.get("augmentation") or {}
    steps = [*_resize(data_cfg)]
    if aug.get("crop_scale"):
        steps.append(
            T.RandomResizedCrop(
                size,
                scale=tuple(aug["crop_scale"]),
                ratio=tuple(aug.get("crop_ratio", (0.8, 1.25))),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    if aug.get("rotation_degrees"):
        steps.append(T.RandomRotation(float(aug["rotation_degrees"]), fill=(124, 116, 104)))
    if aug.get("horizontal_flip_prob"):
        steps.append(T.RandomHorizontalFlip(float(aug["horizontal_flip_prob"])))
    jitter = aug.get("color_jitter")
    if jitter and aug.get("color_jitter_prob", 0) > 0:
        steps.append(T.RandomApply([T.ColorJitter(**jitter)], p=float(aug["color_jitter_prob"])))
    if aug.get("grayscale_prob"):
        steps.append(T.RandomGrayscale(float(aug["grayscale_prob"])))
    if aug.get("blur_prob"):
        steps.append(T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=float(aug["blur_prob"])))
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)
