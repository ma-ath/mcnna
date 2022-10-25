# Monte Carlo CNN Vision Analisys

An approach to try to understand what a CNN is looking at when classifying images using a monte carlo based algorithnm.

![](/assets/dog_readme.png "Output of MCNNA")

## How it works

Given a **CNN** trained model and an image input, we run the following simplified algorithnm:

    1.  Create a set of random maps.
    2.  Apply each map as a mask to the input. Observe how the output changes for each masked input.
    3.  Calculate a set of Attention Maps for each label, with a weighted sum of each random map and the output distance between the masked-input and unmasked-input outputs.
    4.  Calculate a PCA of all Attention Maps.

## Results
The PCA output will show the main components of all attention maps for each class, togheter with their importance (given by its singular value).

# Package
The class MCNNA (*monte-carlo neural network analisys*) is provided as a simple package for the algorithm.

## Example code
```
from MCNNA import MCNNA
from torchvision.io import read_image
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import ToPILImage
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load an image as a torch tensor
image = read_image(str(Path('assets') / 'cat_resize.jpg'))

# Load a model, its weights and input transforms
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights, progress=False)
transforms = weights.transforms()

# Initiate object with 1000 random maps and (224,224) input shape
mcnna = MCNNA(1000, (224,224))

# Run the simulation 
mcnna.simulate(image, model, transforms)

# Run the PCA analisys
mcnna.pca()

# Show the PCA Analisys
fig, axes = plt.subplots(2, 5, figsize=(1.5*5,2*2))
for i in range(10):
    ax = axes[i//5, i%5]
    img = ToPILImage()(image.cpu())
    ax.imshow(np.asarray(img), cmap='gray')
    ax.set_title(f'σ={mcnna.S[i]:.2f}')
    ax.imshow(mcnna.attention_maps_pca[i], cmap='jet', alpha=0.5)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

A [Jupyter notebook example](jupyter-example.ipynb) is also available

# Full algorithnm:
>   ## Generation of random maps
>   1. Generate a set of $n$ "blob" random maps *r_maps* between [0,1].
>
>       1.1 Generate a random gaussian noise.
>
>       1.2 Apply a Gaussian blur.
>
>       1.3 Adjust exposure.
>
>       1.4 Apply a threshold filter.
>
>       1.5 Smooth shapes with a morphology transform.
>
>   2.  Calculate the mean random map *mean_map*.
>   ## Monte-carlo Simulation
>   3.  Calculate the CNN vector output for the given image without any mask $\text{nomask output}$.
>
>   4.  For each *r_maps*, apply the masking (point-wise multiplication) to the image and calculate the CNN vector output for the given mask $\text{mask output}$.
>
>   ## Calculation of Attention maps
>   5.  For each position $i$ of the CNN output, calculate an **attention map** with a weighted sum of each $(\text{r-map} - \text{mean-map})$ by the distance $(\text{mask output} - \text{nomask output})$ at position $i$.
>   ## (Optional) Calculation of Attention map using Euclidian Norm
>   6. Calculate an **euclidian attention map** with a weighted sum of each $(\text{r-map} - \text{mean-map})$ by the euclidian distance of $(\text{mask output} - \text{nomask output})$.
>   ## Calculate the PCA of all Attention Maps
>   7. Construct a column matrix $X$ of **attention maps**
>
>       7.1 Normalize all columnsby removing each column mean and dividing by $\sqrt{\text{data size}}$
>   8. Calculate the (reduced) SVD decomposition $U \Sigma V^T$
>   
>   9. Return eigenmaps in the columns of $U$