from pathlib import Path
import numpy as np
import yolov5

# load model
model = yolov5.load('keremberke/yolov5n-license-plate')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = str(Path("input/car2.png"))

# perform inference
results = model(img, size=640)

# inference with test time augmentation
results = model(img, augment=True)

for i in range(len(results.ims)):
    if not results.ims[i].flags.writeable:
        results.ims[i] = np.array(results.ims[i]).copy()
        results.ims[i].setflags(write=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')
