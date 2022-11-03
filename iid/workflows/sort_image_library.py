from pathlib import Path
from typing import List

import prefect

from iid.model.detection import detect_objects
from iid.processing.postprocessing import move_detections_to_output
from iid.processing.preprocessing import convert_images_to_tensors, read_images
from iid.system_interaction import generate_batches, image_discovery


@prefect.flow(task_runner=prefect.task_runners.ConcurrentTaskRunner())
def coordinate_batching(
    image_directory: str = "images",  # "/Users/owrior/Downloads/Liz stripe I.D",
    output_directory: str = "output_images",
    batch_size: int = 1000,
    draw_boxes: bool = True,
):
    all_images = image_discovery(image_directory)
    image_batches = generate_batches(all_images, batch_size)

    for image_batch in image_batches:
        process_batch(image_batch, output_directory, draw_boxes)


@prefect.flow(task_runner=prefect.task_runners.ConcurrentTaskRunner())
def process_batch(image_batch: List[Path], output_directory: str, draw_boxes: bool):
    images = read_images(image_batch)
    image_tensors = convert_images_to_tensors(images)
    predictions = detect_objects(image_tensors)
    move_detections_to_output(
        images, image_batch, predictions, output_directory, draw_boxes
    )


if __name__ == "__main__":
    coordinate_batching()
