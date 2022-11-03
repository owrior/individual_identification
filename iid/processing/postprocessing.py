import shutil
from pathlib import Path
from typing import List

import prefect
import torch
from PIL import Image, ImageDraw


@prefect.task
def move_detections_to_output(
    images: List[Image.Image],
    image_paths: List[Path],
    predictions: torch.tensor,
    image_directory: str,
    output_directory: str,
    draw_boxes: bool,
) -> None:
    """
    Move images with a detected score above a specified threshold to the output
    directory.
    """
    shutil.copytree(
        image_directory,
        output_directory,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.jpg", "*.jpeg", "*.JPG", "JPEG"),
    )
    for _, detections in enumerate(predictions):
        image_path = image_paths[_]
        if (detections["scores"] > 0.7).any():
            write_location = Path(output_directory) / image_path.relative_to(
                image_directory
            )
            if draw_boxes:
                image = images[_].copy()
                for score, box in zip(detections["scores"], detections["boxes"]):
                    (startX, startY, endX, endY) = (
                        box.detach().cpu().numpy().astype("int")
                    )
                    img_draw = ImageDraw.Draw(image)
                    img_draw.rectangle([(startX, startY), (endX, endY)], outline="red")
                    y = startY - 15 if startY > 30 else startY + 15
                    img_draw.text((startX, y), f"{torch.round(score, decimals=4)}")
                image.save(write_location)
            else:
                write_location = Path(output_directory) / image_path.relative_to(
                    *image_path.parts[:1]
                )
                shutil.copyfile(image_path, write_location)
