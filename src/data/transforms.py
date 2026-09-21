"""
SSL and evaluation augmentation pipelines for MI-JEPA.
"""
from lightly.transforms import SimCLRViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torchvision import transforms


def get_ssl_transform(img_size: int = 224):
    """Heavy augmentation for SSL training (both inter and intra bags).

    Each bag slot is augmented independently, so this is the *single-view*
    SimCLR pipeline (`SimCLRTransform` would return a list of two views and
    break the one-crop-per-slot contract of the bag collate function).
    """
    return SimCLRViewTransform(
        input_size=img_size,
        min_scale=0.2,
        cj_prob=0.8,
        cj_strength=1.0,
        cj_bright=0.4,
        cj_contrast=0.4,
        cj_sat=0.4,
        cj_hue=0.1,
        random_gray_scale=0.2,
        gaussian_blur=0.5,
        sigmas=(0.1, 2.0),
        hf_prob=0.5,
        normalize=IMAGENET_NORMALIZE,
    )


def get_eval_transform(img_size: int = 224):
    """Deterministic transform for evaluation / feature extraction."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_NORMALIZE),
    ])
