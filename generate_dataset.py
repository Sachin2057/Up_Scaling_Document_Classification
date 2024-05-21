import os
import shutil
import random
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
import config


def get_main_augmentation():
    """
    Creates and returns a spatial augmentation pipeline for main data augmentation.

    The augmentation pipeline includes the following transformations:
    - RandomResizedCrop: Randomly crops and resizes the image to 224x224 pixels,
                         with a scale between 80% and 100% of the original size.
    - HorizontalFlip: Horizontally flips the image with a 50% probability.
    - VerticalFlip: Vertically flips the image with a 50% probability.
    - Rotate: Rotates the image within a range of -30 to +30 degrees with a 50% probability.
    - Affine: Applies an affine transformation with a shear range of -25 to +25 degrees and a 50% probability.

    Returns
    -------
    A.Compose
        An Albumentations Compose object that applies the defined transformations.
    """
    transform = A.Compose(
        [
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Affine(shear=[-25, 25], p=0.5),
        ]
    )
    return transform


def get_valid_augmentation():
    """
    Creates and returns a color jitter augmentation pipeline for validation data.

    The augmentation pipeline includes the following transformations:
    - ColorJitter (brightness): Adjusts brightness by a factor of 0.8 to 1.2.
    - ColorJitter (contrast): Adjusts contrast by a factor of 0.5 to 1.5.
    - ColorJitter (saturation): Adjusts saturation by a factor of 0.5 to 1.5.
    - ColorJitter (hue): Adjusts hue by a factor of 0.5.

    Returns
    -------
    A.Compose
        An Albumentations Compose object that applies the defined transformations.
    """
    valid_transform = A.Compose(
        [
            A.ColorJitter(brightness=(0.8, 1.2), contrast=0, saturation=0, hue=0),
            A.ColorJitter(brightness=0, contrast=(0.5, 1.5), saturation=0, hue=0),
            A.ColorJitter(brightness=0, contrast=0, saturation=(0.5, 1.5), hue=0),
            A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
        ]
    )
    return valid_transform


def image_augmentation(image_path, transformation):
    """
    Performs image augmentation using the specified transformation.

    Parameters
    ----------
    image_path : str or ndarray
        Path to the image file or the image array.
    transformation : A.Compose
        The Albumentations transformation to be applied.

    Returns
    -------
    numpy.ndarray
        The augmented image.
    """
    if type(image_path) is str:
        image = cv2.imread(image_path)
    else:
        image = image_path
    augmented = transformation(image=image)
    augmented_image = augmented["image"]
    return augmented_image


def generate_dataset():
    """
    Loads images from the raw folders, splits them into training and validation sets,
    applies augmentations, and saves the augmented images to the corresponding directories.

    The function performs the following steps:
    - For each class directory, lists all image files.
    - Splits the image files into training and validation sets.
    - For the training set, applies the main augmentation and saves the augmented images.
    - For the validation set, applies the main and validation augmentations and saves the augmented images.

    Returns
    -------
    None
    """
    for ix, class_dir in enumerate(config.classes):
        files = os.listdir(class_dir)
        # num_samples_to_augment = int(0.4 * len(files))
        # indices_to_augment = random.sample(range(len(files)), num_samples_to_augment)
        # for index in indices_to_augment:
        #     image_path = os.path.join(class_dir, files[index])
        #     transforamtion = get_main_augmentation()
        #     augmented_image = image_augmentation(
        #         image_path=image_path, transformation=transforamtion
        #     )
        #     save_path = os.path.join(class_dir, f"{files[index]}_augmented.jpg")
        #     cv2.imwrite(save_path, augmented_image)
        if len(files) != 0:
            train, test = train_test_split(files, test_size=0.2)
            for image in train:
                orginal_path = os.path.join(class_dir, image)
                transforamtion = get_main_augmentation()
                augmented_image = image_augmentation(
                    image_path=orginal_path, transformation=transforamtion
                )
                save_path_orginal = os.path.join(
                    config.TRAIN_SAVE_DIRECTORY, f"{ix}-{image}.jpg"
                )
                save_path_augmented = os.path.join(
                    config.TRAIN_SAVE_DIRECTORY, f"{ix}-{image}_augmented.jpg"
                )
                cv2.imwrite(save_path_augmented, augmented_image)
                shutil.move(orginal_path, save_path_orginal)
            for image in test:
                orginal_path = os.path.join(class_dir, image)
                main_transforamtion = get_main_augmentation()
                transforamtion = get_valid_augmentation()
                augmented_image = image_augmentation(
                    orginal_path, transformation=main_transforamtion
                )
                augmented_image = image_augmentation(
                    augmented_image, transformation=transforamtion
                )
                save_path = os.path.join(
                    config.VALID_SAVE_DIRECTORY, f"{ix}-{image}.jpg"
                )
                save_augmented_path = os.path.join(
                    config.VALID_SAVE_DIRECTORY, f"{ix}-{image}_augmented.jpg"
                )
                cv2.imwrite(save_augmented_path, augmented_image)
                shutil.move(orginal_path, save_path)
