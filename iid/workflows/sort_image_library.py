from pathlib import Path
from typing import List

from iid.model.detection import detect_objects
from iid.processing.postprocessing import (
    create_destination_structure,
    move_detections_to_output,
)
from iid.processing.preprocessing import convert_images_to_tensors, read_images
from iid.system_interaction import generate_batches, image_discovery


def coordinate_batching(
    image_directory: str = "/Users/owrior/Downloads/Camera Trap Images",
    output_directory: str = "output_images",
    batch_size: int = 50,
    tolerance: float = 0.3,
    draw_boxes: bool = False,
):
    create_destination_structure(image_directory, output_directory)

    all_images = image_discovery(image_directory)
    image_batches = generate_batches(all_images, batch_size)

    for image_batch in image_batches:
        process_batch(
            image_batch, image_directory, output_directory, tolerance, draw_boxes
        )


def process_batch(
    image_batch: List[Path],
    image_directory: str,
    output_directory: str,
    tolerance: float,
    draw_boxes: bool,
):
    images = read_images(image_batch)
    image_tensors = convert_images_to_tensors(images)
    predictions = detect_objects(image_tensors)
    move_detections_to_output(
        images,
        image_batch,
        predictions,
        image_directory,
        output_directory,
        tolerance,
        draw_boxes,
    )


if __name__ == "__main__":
    coordinate_batching()
