import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# =====================================================
# Build transforms from config
# =====================================================
def build_transforms(aug):

    # -------------------------
    # Train transforms
    # -------------------------
    if aug == "basic":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    elif aug == "strong":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
        ])

    else:  # fallback
        transform_train = transforms.ToTensor()

    # -------------------------
    # Test transforms
    # -------------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    return transform_train, transform_test


# =====================================================
# Dataloader
# =====================================================
def get_dataloader(config):
    aug = config.get("augmentation", "basic")
    transform_train, transform_test = build_transforms(aug)

    batch_size = config.get("batch_size", 128)

    num_workers = config.get("num_workers", 4)

    # -------------------------
    # Datasets
    # -------------------------
    trainset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    # -------------------------
    # Loaders
    # -------------------------
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainloader, testloader