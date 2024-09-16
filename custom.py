import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = torchvision.datasets.CocoDetection(root=root, annFile=annotation_file)

    def __getitem__(self, idx):
        img, target = self.coco[idx]
        img = F.to_tensor(img)

        boxes = []
        labels = []
        for annotation in target:
            xmin, ymin, width, height = annotation['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(annotation['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(target),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco)
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item()}")
        i += 1

    lr_scheduler.step()

    print(f"Epoch: {epoch}, Training Loss: {losses.item()}")
    
    
   
