import os
import shutil
import random
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
import config


def get_main_augmentation():
    """
    Saptial augmentation for main data augmentation
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
    Validation augmentation
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
    Perform image augmentation

    Parameters
    ----------
    image_path : str or nd_array
        path of the image
    transformation : transfomation
        Transformation to be applied

    Returns
    -------
    image:numpy_array
        Augmented image
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
    Load images from the raw folders split the images into train split
    add augmentaion and save in training directory
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
