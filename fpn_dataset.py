from os import path, listdir
import torch
from torchvision import transforms
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


colors_per_class = {
    'excluded' : [255, 0, 0],
    'selected' : [0, 0, 255],
    'p3' :     [0, 255, 255],
    'p2' :     [0, 255, 0],
}
# processes fpn dataset
class FPNDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, folder, num_images=1000):
        translation = {'p0' : 'all',
                       'rgb' : '100',}
        self.classes = translation.values()

        if not path.exists(data_path):
            raise Exception(data_path + ' does not exist!')

        self.data = []

        label = folder

        full_path = path.join(data_path, folder)
        images = listdir(full_path)

        current_data = [(path.join(full_path, image), label) for image in images]
        self.data += current_data

        num_images = min(num_images, len(self.data))
        self.data = random.sample(self.data, num_images) # only use num_images images

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert('RGB')
        trans=transforms.ToTensor()
        image=trans(image)
        #try:
        #    image = self.transform(image) # some images in the dataset cannot be processed - we'll skip them
        #except Exception:
        #    return None

        dict_data = {
            'image' : image,
            'label' : label,
            'image_path' : image_path
        }
        return dict_data

# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)
