from typing import Dict, List

import torch
from torchvision.models import detection


def detect_objects(
    image_tensors: torch.Tensor,
    model=detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    ),
) -> List[Dict[str, torch.tensor]]:
    """
    Detects objects and predicts their classification with a specified model.
    """
    model.eval()
    return model(image_tensors)
