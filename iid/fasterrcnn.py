from pathlib import Path

import torchvision
from PIL import Image, ImageDraw

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
img_to_tensor = torchvision.transforms.ToTensor()
tensor_to_image = torchvision.transforms.ToPILImage()

image_path = Path("images")
detected_path = Path("output_images")
CROP = False

# For inference
model.eval()

image_locations = list(image_path.glob("*.jpeg"))
images = [Image.open(image) for image in image_locations]

x = [img_to_tensor(image) for image in images]
predictions = model(x)


for _, detections in enumerate(predictions):
    # loop over the detections
    for i in range(len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.7:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box and label on the image
            orig = images[_].copy()
            img_draw = ImageDraw.Draw(orig)
            img_draw.rectangle([(startX, startY), (endX, endY)], outline="red")
            y = startY - 15 if startY > 30 else startY + 15
            img_draw.text((startX, y), f"{confidence}")
            if CROP:
                orig = orig.crop()
            else:
                orig.save(detected_path / image_locations[_].name)
