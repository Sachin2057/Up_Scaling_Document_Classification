import os

DATASET_DIRECTORY = "Raw"
CITIZENSHIP_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Citizenship")
PASSPORT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Passport")
OTHER_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Others")
LICENSE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "License")


TRAIN_SAVE_DIRECTORY = os.path.join("Dataset", "Train")
VALID_SAVE_DIRECTORY = os.path.join("Dataset", "Valid")

classes = [
    CITIZENSHIP_DIRECTORY,
    PASSPORT_DIRECTORY,
    LICENSE_DIRECTORY,
    OTHER_DIRECTORY,
]

######Model parameters
batch_size = 16
classes_name = ["Citizenship", "Passport", "License", "Others"]
num_classes = 4
