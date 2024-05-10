import cv2
import torch
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.transforms import transforms
from py_modules.model import ClassificationModel
import config
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_grad_cam(model, input_image_path, target):
    """
    Generate Grad-CAM visualization for a given input image and target class.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which Grad-CAM is generated.
    input_image_path : str or np.ndarray
         Path to the input image file or
            numpy array representing the input image
    target : int
        Target class index for which Grad-CAM is computed.

    Returns
    -------
    np.ndarray
         Grad-CAM visualization overlaid on the input image.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preprocessing = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if type(input_image_path) is str:
        image = cv2.imread(input_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.float32(image) / 255

    else:
        image = input_image_path
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = np.float32(image) / 255

    image_tensor = preprocessing(image).to(device)
    image_tensor = image_tensor[None, :, :, :]
    # model = ClassificationModel(num_classes=config.num_classes).to(device)
    # model.load_state_dict(torch.load(checkpoint))
    target_layer = model.resnet.layer4
    # cam = GradCAM(model=model, target_layers=target_layer)
    targets = [
        ClassifierOutputTarget(target),
    ]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cams = cam(input_tensor=image_tensor, targets=targets)
        cam_image = show_cam_on_image(image, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255 * grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((np.uint8(255 * image), cam, cam_image))
        images = Image.fromarray(images)
        images = np.array(images)
        return images
