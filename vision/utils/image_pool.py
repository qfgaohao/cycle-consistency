import torch
import random


class ImagePool:

    def __init__(self, size=50):
        self.size = size
        self.pool = []

    def query(self, images):
        """Randomly replace the images with images in the pool."""
        if self.size <= 0:
            return images

        results = []
        for image in images:
            if len(self.pool) < self.size:
                self.pool.append(image)
                results.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    selected_index = random.randint(0, len(self.pool) - 1)
                    results.append(self.pool[selected_index])
                    self.pool[selected_index] = image
                else:
                    results.append(image)
        results = torch.stack(results, 0)
        return results