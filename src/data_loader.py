import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=256):  # 增大batch_size到256
    # 增强的数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),  # 减小padding强度
        transforms.RandomHorizontalFlip(p=0.5),  # 明确设置翻转概率
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # 加载 Fashion-MNIST 数据集
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform_test
    )

    # 划分训练集和验证集（90% 训练，10% 验证）
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 增大num_workers提高数据加载速度
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader