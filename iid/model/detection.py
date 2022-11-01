from typing import List

import prefect
import torchvision


@prefect.task
def detect_objects(
    image_tensors: List[torchvision.torch.Tensor],
    model=torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        pretrained=True
    ),
):
    """
    Detects objects and predicts their classification with a specified model.
    """
    return model(image_tensors)
