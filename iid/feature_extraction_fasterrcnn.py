import torchvision
import torch

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

# Freeze model gradients
for param in model.parameters():
    param.requires_grad = False

# Unfreeze box
for param in model.roi_heads.box_predictor.bbox_pred.parameters():
    param.requires_grad = True

model.eval()
