"""
Inference for classification model
Returns
-------
_type_
    _description_
"""

import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, image):
    """
    Predict the class label and probabilities for a given image using a trained model.s

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model for classification.
    image : PIL.Image
        The input image to be classified.


    Returns
    -------
     Tuple[int, float]
         A tuple containing the predicted class label and the probability of the prediction.
    Note:
        This function assumes that the model is trained for image classification tasks.
        It applies a series of transformations to the input image, including resizing, normalization, and conversion to a tensor.
        The prediction is made using the provided model, and the class label with the highest probability is returned.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image).to(device)
    image_tensor = image_tensor[None, :, :, :]
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    print(output)
    probabilities = F.softmax(output, dim=1)[0]
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item(), probabilities[predicted_class].item()
