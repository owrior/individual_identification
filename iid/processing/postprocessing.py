import shutil
from pathlib import Path

import prefect


@prefect.task
def move_detections_to_output(image_paths, predictions, output_directory):
    """
    Move images with a detected score above a specified threshold to the output
    directory.
    """
    for _, detections in enumerate(predictions):
        if (detections["scores"] > 0.7).any():
            shutil.copy(image_paths[_], Path(output_directory) / image_paths[_])
