from typing import List

import prefect

from iid.model.detection import detect_objects
from iid.processing.postprocessing import move_detections_to_output
from iid.processing.preprocessing import convert_images_to_tensors, read_images
from iid.system_interaction import generate_batches, image_discovery


@prefect.flow(task_runner=prefect.task_runners.SequentialTaskRunner())
def coordinate_batching(
    image_directory: str = "images",
    output_directory: str = "output_images",
    batch_size: int = 1000,
):
    all_images = image_discovery(image_directory)
    image_batches = generate_batches(all_images, batch_size)

    for image_batch in image_batches:
        process_batch(image_batch, output_directory)


@prefect.flow(task_runner=prefect.task_runners.SequentialTaskRunner())
def process_batch(image_batch: List[str], output_directory: str):
    images = read_images(image_batch)
    image_tensors = convert_images_to_tensors(images)
    predictions = detect_objects(image_tensors)
    move_detections_to_output(image_batch, predictions, output_directory)
