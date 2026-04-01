"""Dataset with aggressive augmentation for contrastive learning."""
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps ImageFolder to return two augmented views of each image for contrastive learning."""
    def __init__(self, root, transform):
        self.ds = datasets.ImageFolder(root, transform=transform)
        self.targets = self.ds.targets
        self.classes = self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Apply transform twice for two views
        img, label = self.ds[idx]
        # Re-apply transform on original PIL image for second view
        img2, _ = self.ds[idx]
        return img, img2, label


def build_loaders(data_root, batch_size=128, num_workers=4):
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_ds = TwoViewDataset(f"{data_root}/train", train_transform)
    test_ds = datasets.ImageFolder(f"{data_root}/test", test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, len(train_ds.classes), train_ds.classes
