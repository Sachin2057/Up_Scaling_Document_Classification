import os
import time
import torch
from torch.utils.data import DataLoader
from py_modules.model import ClassificationModel
import torch.nn as nn
import torch.optim
from py_modules.utils.utils import generate_confusion_matrix, save_checkpoint

import config
from tensorboardX import SummaryWriter
from py_modules.dataloader import ImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"


def training():
    """
    Train a classification model.

    This function performs training for a classification model using the provided dataset and configuration.

    Returns:
        None: Training results are logged and saved to disk.

    Note:
        This function assumes the presence of a dataset in the 'Dataset' directory.
        It utilizes a custom ImageClassification dataset class for loading data.
        The training process includes logging training loss, saving checkpoints, evaluating validation loss and accuracy,
        and generating a confusion matrix.
        Checkpoints, logs, and confusion matrix are saved in separate directories for organization.
    """

    train_dataset = ImageClassification(root_directory="Dataset", train=True)
    valid_dataset = ImageClassification(root_directory="Dataset", train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    EPOCH = 40
    start_time = time.strftime("%b-%d_%H-%M-%S")
    model = ClassificationModel(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    log_path = os.path.join("logs", f"{start_time}")
    writer = SummaryWriter(log_dir=log_path)
    for i in range(EPOCH):
        print(EPOCH)
        model.train()
        running_train_loss = 0.0
        for image, target in train_dataloader:
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        writer.add_scalar(
            "Training loss", running_train_loss / len(train_dataloader), i
        )
        save_checkpoint(
            model=model, optimizer=optimizer, start_time=start_time, epoch=i
        )
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for image, target in valid_dataloader:
                output = model(image)
                loss = criterion(output, target)
                running_val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
            val_loss = running_val_loss / len(valid_dataloader)
            val_accuracy = correct / total * 100
            writer.add_scalar("validation_loss", val_loss, i)
            writer.add_scalar("validation_accuracy", val_accuracy, i)

    generate_confusion_matrix(
        model=model, writer=writer, valid_dataloader=valid_dataloader
    )
