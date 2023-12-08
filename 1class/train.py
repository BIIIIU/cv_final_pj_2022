from ultralytics import YOLO
from PIL import Image
import argparse
# Load a model
model = YOLO('/project/yolov8n-pose.pt')  # load a pretrained model (recommended for training)

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="/project/runs/1class")
parser.add_argument("--data", default = "/project/train/src_repo/1class_2/test.yaml")
# parser.add_argument("--save_dir", default = "/project/rus/1class/results")

args = parser.parse_args()
project_value = args.project
data_value = args.data
# save_dir = args.save_dir

import torch
print(torch.cuda.device_count())

# Train the modelx
results = model.train(data=data_value, epochs=80, imgsz=1920, batch = 4, project = project_value, device = 0)#

results = model('/project/data/street_1_015529.jpg')
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('results.jpg')

