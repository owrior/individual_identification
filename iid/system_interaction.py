import logging
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_PATTERNS = ["*.jpg", "*.jpeg", "*.JPG", ".*JPEG"]


def image_discovery(image_directory: str) -> List[str]:
    """
    Fetch image paths either from a specified location or the default.
    """

    image_paths = []
    for image_pattern in IMAGE_PATTERNS:
        image_paths += list(Path(image_directory).rglob(image_pattern))
    logger.info(f"{len(image_paths)} images found.")
    return image_paths


def generate_batches(image_paths: List[str], batch_size: int) -> List[List[Path]]:
    """
    Split list of all images into batches.
    """
    num_images = len(image_paths)
    batches = [
        image_paths[i : i + batch_size] for i in range(0, num_images, batch_size)
    ]
    logger.info(f"Split into {len(batches)} batches of  maximum size {batch_size}")
    return batches
