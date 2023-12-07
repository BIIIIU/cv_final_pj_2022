import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, image_folder, keypoint_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.keypoint_folder = keypoint_folder
        self.mask_folder = mask_folder
        self.transform = transform

        # Get only filenames with .jpg extension
        self.filenames = [filename for filename in os.listdir(image_folder) if filename.endswith('.jpg')]


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_folder, self.filenames[idx])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transforms.ToTensor()(image)

        # Load keypoint information
        keypoint_path = os.path.join(self.keypoint_folder, self.filenames[idx].replace('.jpg', '.json'))
        with open(keypoint_path, 'r') as f:
            keypoint_data = json.load(f)

        # Load mask information
        mask_path = os.path.join(self.mask_folder, self.filenames[idx].replace('.jpg', '.png'))
        # mask = np.load(mask_path)
        mask = Image.open(mask_path)
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = transforms.ToTensor()(mask).squeeze()
        
        data = []

        for annotation in keypoint_data['annotations']:
            image_id = annotation['image_id']
            keypoints = annotation['keypoints']

            class_name = annotation['category_name']
            if(class_name == "person"):
                category_id = 0
            else:
                category_id = 1
            
            yolo_line = []

            # 添加关键点信息
            for i in range(0, len(keypoints), 3):
                px, py, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                yolo_line.append(px)
                yolo_line.append(py)
                yolo_line.append(visibility)
            # print(yolo_line)
            # exit()
            keypoint_tensor = torch.tensor(yolo_line)
            keypoint_tensor = keypoint_tensor.reshape(-1)
            data.append({'image': image_tensor, 'keypoints': keypoint_tensor, 'mask': mask_tensor})
            # print(type(data[0]['keypoints']))
            # exit()
        # Apply transformations if specified
        # if self.transform:
        #     data = self.transform(data)
        # print("type data:", type(data[0]))
        return data

# Example usage
transform = transforms.Compose([
    # Add your desired transformations here
])

# Replace 'your_image_folder', 'your_keypoint_folder', and 'your_mask_folder' with your actual folder paths
dataset = CustomDataset(image_folder='/project/data', keypoint_folder='/project/data',
                        mask_folder='/project/data', transform=transform)

# Adjust batch_size, shuffle, and other DataLoader parameters based on your requirements
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through the dataloader to get batches of data
for batch in dataloader:
    print(len(batch))
    print(batch[0])
    exit()
