from torchvision.transforms import Compose, Normalize, RandomApply, ColorJitter, RandomGrayscale, RandomResizedCrop, RandomHorizontalFlip, ToTensor


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)

transforms_train = Compose([
    RandomResizedCrop((32,32)),
    RandomHorizontalFlip(p=0.5),
    RandomApply([
        ColorJitter(0.8, 0.8, 0.8, 0.2),
    ], p=0.8),
    RandomGrayscale(p=0.2),
    ToTensor(),
#    Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

transforms_valid = transforms_train

transforms_test = Compose([
    ToTensor(),
#    Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
