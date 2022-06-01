from torchvision import transforms
from .randaug import RandAugment
from PIL import Image
import PIL


def mae_transform(size):
    trs = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return trs

def mae_transform_randaug(size):
    trs = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=3),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def mae_transform_test(size):
    t = []
    t.append(
        transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t)


