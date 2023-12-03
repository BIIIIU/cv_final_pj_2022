import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import argparse
import cv2
# Load a model
model = YOLO('/project/yolov8n-pose.pt')

def init():
    """Initialize model
    Returns: model
    """
    # model = YOLO('/project/train/models/train_save/train/weights/best.pt')
    model = YOLO('/project/train/models/train0/train/weights/best.pt')
    return model

def process_image(handle=None, input_image=None, args=None, ** kwargs):
    """
        Do inference to analysis input_image and get output
            Attributes:
            handle: algorithm handle returned by init()
            input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
            args: string in JSON format, format: {
                "mask_output_path": "/path/to/output/mask.png"
            }
        Returns: process result
    """
    # Process image here
    # args = json.loads(args)
    # mask_output_path = args['mask_output_path']
    mask_output_path = "/project/mask.png"
    # Generate dummy mask data
    h, w, _ = input_image.shape

    # num_class是指类别个数，包含背景类，并且背景类在第0个位置
    dummy_data = np.random.randint(2, size=(w, h), dtype=np.uint8)
    pred_mask_per_frame = Image.fromarray(dummy_data)
    pred_mask_per_frame.save(mask_output_path)

    result = model(input_image, imgsz = [h, w])
    # print(result[0].Boxes)   # Boxes object for bbox outputs
    # print(result[0].masks.shape)  # Masks object for segmentation masks outputs
    # print(result[0].keypoints.shape)  # Keypoints object for pose outputs
    # print(result[0].probs.shape)

    result_json = {
        "algorithm_data": {
            "is_alert": True,
            "target_count": 0,
            "target_info": []
        },
        "model_data": {"mask": "mask_output_path", "objects": []}
    }
    result = result[0]
    for i in range(len(result.boxes)):
        detection = {
            "x": result.boxes[i].xywh[0],
            "y": result.boxes[i].xywh[1],
            "width": result.boxes[i].xywh[2],
            "height": result.boxes[i].xywh[3],
            "confidence": result.boxes[i].conf(),
            "name": "person",  # 这里是硬编码的示例值，你可能需要根据你的实际情况修改
            "keypoints": {
                "keypoints": result["keypoints"][i],
                "score": result["probs"][i]  # 假设关键点的分数与检测的置信度相同
            }
        }
        result_json["model_data"]["objects"].append(detection)



    fake_result = {}
    
    fake_result["algorithm_data"] = {
        "is_alert": True,
        "target_count": 0,
        "target_info": []
    }
    fake_result["model_data"] = {
        "mask": "mask_output_path",
        "objects": [
            {
                "x": 1805,
                "y": 886,
                "width": 468,
                "height": 595,
                "confidence": 0.7937217950820923,
                "name": "person",
                "keypoints": {
                    "keypoints": [
                        2161.423828125,
                        990.58984375,
                        1.0,
                        2161.423828125,
                        981.29296875,
                        1.0,
                        2161.423828125,
                        981.29296875,
                        1.0,
                        2093.238525390625,
                        967.34765625,
                        1.0,
                        2124.231689453125,
                        985.94140625,
                        1.0,
                        2031.251708984375,
                        995.23828125,
                        1.0,
                        2068.443603515625,
                        1069.61328125,
                        1.0,
                        2093.238525390625,
                        1074.26171875,
                        1.0,
                        2124.231689453125,
                        1190.47265625,
                        1.0,
                        2161.423828125,
                        1088.20703125,
                        1.0,
                        2198.615966796875,
                        1130.04296875,
                        1.0,
                        1944.47021484375,
                        1185.82421875,
                        1.0,
                        2012.6556396484375,
                        1236.95703125,
                        1.0,
                        2124.231689453125,
                        1106.80078125,
                        0.0,
                        2186.218505859375,
                        1195.12109375,
                        1.0,
                        2130.430419921875,
                        1232.30859375,
                        0.0,
                        2173.8212890625,
                        1246.25390625,
                        1.0
                    ],
                    "score": 0.7535077333450317
                }
            }
        ]
    }
    return json.dumps(fake_result, indent=4)

if __name__ == '__main__':
    img = cv2.imread('/project/data/street_1_015529.jpg')
    predictor = init()
    import time
    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print((e-s))