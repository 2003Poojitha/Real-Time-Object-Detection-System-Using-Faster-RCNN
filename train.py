import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from torch.utils.data import DataLoader

root = 'path/to/images'
annotation_file = 'path/to/annotations.json'


dataset = CustomDataset(root, annotation_file, get_transform(train=True))
dataset_test = CustomDataset(root, annotation_file, get_transform(train=False))

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
