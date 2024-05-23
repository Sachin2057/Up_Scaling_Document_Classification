import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import config
from torchmetrics import F1Score


def generate_confusion_matrix(model, writer, valid_dataloader, model_name):
    """
    Generates confiusion matrix

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to evaluate
    writer (torch.utils.tensorboard.SummaryWriter):

    valid_dataloader :(torch.utils.data.DataLoader)
        The validation dataloader
    """
    all_preds = []
    all_labels = []
    model.eval()
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for image, target in valid_dataloader:
            if model_name == "ResNet" or model_name == "Inception":
                output = model(image)
            else:
                output, _ = model(image)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)
    writer.add_figure(
        "confusion_matrix", plot_confusion_matrix(conf_matrix, config.classes_name)
    )


def plot_confusion_matrix(cm, classes):
    """
    Return heat map ofconfusion matrix

    Parameters
    ----------
    cm : nd-array
        Confusion matrix
    classes : list
        classes

    Returns
    -------
     matplotlib.figure.Figure
        The Figure object containing the plotted confusion matrix heatmap.

    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def save_checkpoint(model, optimizer, start_time, epoch):
    """
    save checkpoint the model

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose state_dict will be saved.
    optimizer : torch.optim.Optimizer
         The optimizer object whose state_dict will be saved.
    start_time : str
        The starting time of the training session, used for organizing checkpoints.
    epoch : int
       The current epoch number.
    """
    target_dir = os.path.join("checkpoints", str(start_time))
    os.makedirs(target_dir, exist_ok=True)
    save_path_model = os.path.join(target_dir, f"model_{epoch}.pth")
    save_path_optimizer = os.path.join(target_dir, f"optimizer_{epoch}.pth")
    torch.save(model.state_dict(), save_path_model)
    torch.save(optimizer.state_dict(), save_path_optimizer)
    print("Model saved.")


def calcualte_f1_score(model, writer, valid_loader, model_name):
    f1_metric = F1Score(task="multiclass", num_classes=4, average="macro")
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            image, targets = batch
            if model_name == "ResNet" or model_name == "Inception":
                output = model(image)
            else:
                output, _ = model(image)
            f1_metric.update(output.softmax(dim=-1), targets)
        f1_score = f1_metric.compute()
        if writer:
            writer.add_scalar("F1_score", f1_score)
        else:
            print(f1_score)
