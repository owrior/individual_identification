import logging
import shutil
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageDraw

from iId.format.font import GIDOLE_FONT
from iId.format.labels import LABELS
from iId.system_interaction import IMAGE_PATTERNS

logger = logging.getLogger(__name__)


def create_destination_structure(image_directory: str, output_directory: str) -> None:
    (Path(output_directory) / "Undetected").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        image_directory,
        output_directory,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*IMAGE_PATTERNS),
    )


def move_detections_to_output(
    images: List[Image.Image],
    image_paths: List[Path],
    predictions: torch.tensor,
    image_directory: str,
    output_directory: str,
    tolerance: float,
    draw_boxes: bool,
) -> None:
    """
    Move images with a detected score above a specified threshold to the output
    directory.
    """
    number_detected = 0
    for _, detections in enumerate(predictions):
        image_path = image_paths[_]
        if (detections["scores"] > tolerance).any():
            number_detected += 1
            write_location = Path(output_directory) / image_path.relative_to(
                image_directory
            )
        else:
            write_location = Path(output_directory) / "Undetected" / image_path.name

        if draw_boxes:
            image = images[_]
            for score, label, box in zip(
                detections["scores"], detections["labels"], detections["boxes"]
            ):
                (startX, startY, endX, endY) = box.detach().cpu().numpy().astype("int")
                img_draw = ImageDraw.Draw(image)

                img_draw.rectangle(
                    [(startX, startY), (endX, endY)], outline="red", width=4
                )

                y = startY - 15 if startY > 30 else startY + 15
                score_text = round(float(score.detach().numpy()), 4)
                img_draw.text(
                    (startX, y),
                    f"{LABELS[float(label.detach().numpy())]}: {score_text}",
                    font=GIDOLE_FONT,
                )
            image.save(write_location)
        else:
            shutil.copyfile(image_path, write_location)

    logger.info(f"{number_detected} detected in batch.")
