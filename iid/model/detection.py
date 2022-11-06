from typing import Dict, List

import torch
from torchvision.models import detection

FASTER_RCNN_MOBILENET = detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
)

FASTER_RCNN_RESNET = detection.fasterrcnn_resnet50_fpn_v2(
    weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
)


def detect_objects(
    image_tensors: torch.Tensor,
    model=FASTER_RCNN_RESNET,
) -> List[Dict[str, torch.tensor]]:
    """
    Detects objects and predicts their classification with a specified model.
    """
    model.eval()
    return model(image_tensors)
