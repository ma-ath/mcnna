import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        self.vgg16 = vgg16(weights=weights, progress=False).eval()
        self.transforms = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.vgg16(x)
            return y_pred