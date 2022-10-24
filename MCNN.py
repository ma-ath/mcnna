import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MCNN:
    def __init__(self, n:int, size:tuple):
        self.r_maps = self.__RandomMaps(n, size)
        self.solution = []

    def simulate(self, input, model, transforms):
        """
            TODO: Deve implementar eval por batch para ficar mais rapido
        """
        self.result = []
        model = model.eval().to(device)
        with torch.no_grad():
            for mask in self.r_maps:
                input = input.to(device)
                if transforms is not None:
                    input = transforms(input)
                masked_input = input * mask
                batch = torch.stack([masked_input])
                output = model(batch)
                self.result.append(output)

    class __RandomMaps:
        """
        Generate random mask maps for the input
        """
        def __init__(self, n:int, size:tuple):
            self.r_maps = []

            # First map is all ones a.k.a no mask applied
            self.r_maps.append(torch.ones(size).to(device))

            # Other maps are mask 'blobs'
            for i in range(n-1):
                # define random seed to change the pattern
                rng = np.random.default_rng()
                # create random noise image
                noise = rng.integers(0, 255, size, np.uint8, True)
                # blur the noise image to control the size
                blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
                # stretch the blurred image to full dynamic range
                stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
                # threshold stretched image to control the size
                thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
                # apply morphology open and close to smooth out shapes
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
                result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

                self.r_maps.append(torch.from_numpy((255-result)/255).long().to(device))
        
            # Calculate mean map
            self.mean_map = torch.zeros(size).to(device)
            for map in self.r_maps:
                self.mean_map += map / len(self.r_maps)

        def __getitem__(self, n:int):
            return self.r_maps[n]

        def __len__(self):
            return len(self.r_maps)
        
        def mean(self):
            return self.mean_map

    class utils:
        def plot_image(image:torch.Tensor, mask:torch.Tensor=None):
            fix, axs = plt.subplots()
            img = torchvision.transforms.ToPILImage()(image.cpu())
            axs.imshow(np.asarray(img), cmap='gray')
            axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if mask is not None:
                axs.imshow(np.asarray(mask.cpu()), cmap='jet', alpha=0.5)
            fix.show()


if __name__ == '__main__':
    from MCNN import MCNN
    from pathlib import Path
    from torchvision.models import vgg16, VGG16_Weights

    image = torchvision.io.read_image(str(Path('assets') / 'kid_dog_adult.jpg'))

    mcnn = MCNN(10, (224,224))

    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights, progress=False)
    transforms = weights.transforms()

    mcnn.simulate(image, model, transforms)

    print(mcnn.r_maps[0])