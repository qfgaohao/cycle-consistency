from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, pil_loader
import os


class SimpleImageFolder(Dataset):

    def __init__(self, root, transform):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.transform = transform
        images = []
        for dir, _, names in sorted(os.walk(self.root)):
            for name in sorted(names):
                path = os.path.join(root, name)
                if is_image_file(path):
                    images.append(path)
        self.images = images

    def __getitem__(self, index):
        image = self.images[index]
        image = pil_loader(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)
