"""
    ImageClassification dataloader

    Returns
    -------
    tuple (image,target)
        
"""

import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageClassification(Dataset):
    """
    Load image and target
    """

    def __init__(self, root_directory, model_name="", train=True):
        if train:
            self._data_path = os.path.join(root_directory, "train")
        else:
            self._data_path = os.path.join(root_directory, "valid")
        self._data = os.listdir(self._data_path)
        if model_name == "Inception":
            self._transform = self.image_transform(image_size=299, train=train)
        else:
            self._transform = self.image_transform(train)
        self._train = train

    def image_transform(self, image_size=224, train=True):
        """
        Defines runtime transformation to images

        Returns
        -------
        Transform object
            Sequence of transformation
        """
        if train:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            preprocessing = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            preprocessing = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        return preprocessing

    def load_image(self, image_path):
        """
        Load image and transform it into tensor

        Parameters
        ----------
        image_path : str
            Path of image

        Returns
        -------
            tensor
            Pytorch tensor(chanel*h*w)
        """
        # image=Image.open(image_path).convert('RGB')
        # # image=np.transpose(image,(2,0,1))
        # image = self._transform(image)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = cv2.resize(image, (299, 299))
        image = self._transform(image)
        return image

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        image_file = self._data[index]
        image_file = image_file
        target = int(image_file.split("-")[0])
        image_path = os.path.join(self._data_path, image_file)
        image = self.load_image(image_path=image_path)
        return image.to(device), torch.tensor(target).to(device)
