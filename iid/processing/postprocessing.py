import shutil
from pathlib import Path
from typing import List

import prefect
import torch


@prefect.task
def move_detections_to_output(
    image_paths: List[Path], predictions: torch.Tensor, output_directory: str
) -> None:
    """
    Move images with a detected score above a specified threshold to the output
    directory.
    """
    for _, detections in enumerate(predictions):
        image_path = image_paths[_]
        if (detections["scores"] > 0.7).any():
            shutil.copy(
                image_path,
                Path(output_directory) / image_path.relative_to(*image_path.parts[0:1]),
            )
