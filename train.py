from ultralytics import YOLO
from PIL import Image
import argparse
# Load a model
model = YOLO('/project/yolov8n-pose.pt')  # load a pretrained model (recommended for training)

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="/project/train/models/train")

parser.add_argument("--data", default = "/project/train/src_repo/test.yaml")
args = parser.parse_args()
project_value = args.project
data_value = args.data

# Train the modelx
results = model.train(data=data_value, epochs=10, imgsz=1920, batch = 4, project = project_value)#

results = model('/project/data/street_1_015529.jpg')
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('results.jpg')

