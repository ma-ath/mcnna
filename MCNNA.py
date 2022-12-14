def is_notebook() -> bool:
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import numpy as np
import math
import cv2
import skimage
import matplotlib.pyplot as plt
import torchvision
import torch

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------- #

class MCNNA:
    def __init__(self, n:int, size:tuple, ommit_progress:bool=False):
        self.size = size
        self.r_maps = self.__RandomMaps(n, size, ommit_progress)
        self.solution = []
        self.ommit_progress = ommit_progress

    def simulate(self, input, model, transforms=None):
        """
            TODO: Deve implementar eval por batch para ficar mais rapido
        """
        # Pass all masks through model...
        self.output_vectors = []
        model = model.to(device)
        with torch.no_grad():
            input = input.to(device)
            if transforms is not None:
                input = transforms(input)
            for mask in tqdm(self.r_maps, desc= "Inputting random masks to model", disable=self.ommit_progress):
                masked_input = input * mask
                batch = torch.stack([masked_input])
                output = model(batch)
                self.output_vectors.append(output)
            
            self.output_vectors = torch.squeeze(torch.stack(self.output_vectors))
      
            # Create an attention map using the distance from each individual label
            self.attention_maps = []

            for label in tqdm(range(self.output_vectors.shape[1]), desc= "Calculating maps results", disable=self.ommit_progress):
                map = torch.sum(
                        torch.div(
                            torch.mul(
                                torch.subtract(self.r_maps.r_maps, self.r_maps.mean_map),
                                torch.subtract(
                                    self.output_vectors[:, label],
                                    self.output_vectors[0, label]
                                    )[:, None, None]
                                ),
                            len(self.r_maps)
                            ), dim=0
                        )
                self.attention_maps.append(map)
            
            # Create and attention map for the full output vector using euclidian distance
            # We could pararelise this but this is fast enougth none the less.
            self.attention_map_euclidian = torch.zeros(self.size).to(device)

            for i in range(len(self.r_maps)):
                self.attention_map_euclidian += (self.r_maps[i]-self.r_maps.mean_map)*torch.sqrt(torch.sum(torch.square(self.output_vectors[i]-self.output_vectors[0])))/len(self.r_maps)

    def pca(self):
        # Calculate PCA of all attention_maps
        # PCA of all those maps

        X = torch.empty(len(self.attention_maps), math.prod(self.size))
        for i in tqdm(range(len(self.attention_maps)), desc= "Preparing for PCA", disable=self.ommit_progress):
            X[i] = self.attention_maps[i].flatten()
            X[i]-= X[i].mean()
            X[i] = torch.div(X[i], math.sqrt(math.prod(self.size)))
        X=X.t()

        # Calculate the reduced SVD of this data
        print(f'Calculating SVD of data...')
        U, self.S, _ = torch.linalg.svd(X, full_matrices=False)

        self.attention_maps_pca = []

        for i in tqdm(range(len(U[0])), desc= "Reshaping PCA maps", disable=self.ommit_progress):
            self.attention_maps_pca.append(torch.reshape(U[:,i], self.size))
        
        if not self.ommit_progress:
            print(f"Done")

    class __RandomMaps:
        """
        Generate random mask maps for the input
        """
        def __init__(self, n:int, size:tuple, ommit_progress:bool=False):
            self.ommit_progress=ommit_progress
            self.r_maps = []

            # First map is all ones a.k.a no mask applied
            self.r_maps.append(torch.ones(size).to(device))

            # Other maps are mask 'blobs'
            for i in tqdm(range(n-1), desc="Generating random masks", disable=self.ommit_progress):
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
            
            self.r_maps = torch.stack(self.r_maps).to(device)
        
            # Calculate mean map
            self.mean_map = torch.sum(torch.div(self.r_maps, self.r_maps.shape[0]), dim=0)

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
    from MCNNA import MCNNA
    from pathlib import Path
    from torchvision.models import vgg16, VGG16_Weights

    image = torchvision.io.read_image(str(Path('assets') / 'kid_dog_adult.jpg'))

    mcnna = MCNNA(100, (224,224))

    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights, progress=False).eval()
    transforms = weights.transforms()

    mcnna.simulate(image, model, transforms)

    mcnna.pca()