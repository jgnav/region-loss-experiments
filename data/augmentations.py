from PIL import Image
from torchvision import transforms

from utils import training as utils


class DataAugmentationiBOT:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        global_crop_size,
        local_crop_size,
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

        self.global_crops_number = global_crops_number
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crop_size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = [self.global_transfo1(image)]
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
