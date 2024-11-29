import numpy as np
import torchvision.transforms as T
from medmnist import BloodMNIST
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import GaussianBlur



normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.ToTensor(),
    normalize
])
val_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])


class SimSiamTransform():
    def __init__(self, image_size):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


def get_bloodmnist_loader(split='train', img_size=128, batch_size=64, shuffle=True, ratio=None):
    if split == 'train':
        train_dataset = BloodMNIST(
            split=split, download=True, transform=train_transform, size=img_size
        )
        if not ratio:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=shuffle
            )
        else:
            num_labeled_samples = int(ratio * len(train_dataset))
            indices = np.random.choice(len(train_dataset), num_labeled_samples, replace=False)
            train_subset = Subset(train_dataset, indices)
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=shuffle
            )

        return train_loader
    
    elif split == 'val':
        val_dataset = BloodMNIST(
            split=split, download=True, transform=val_transform, size=img_size
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=shuffle)
        return val_loader
    
    elif split == 'test':
        test_dataset = BloodMNIST(
            split=split, download=True, transform=val_transform, size=img_size
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=shuffle)
        return test_loader

    
def get_simsiam_loader(img_size=128, batch_size=64, shuffle=True):
    train_dataset = BloodMNIST(
        split='train', download=True, transform=SimSiamTransform(224), size=img_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True
    )
    return train_loader
    


