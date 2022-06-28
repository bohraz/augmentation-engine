# Create Custom Transformation funtion compatible with PyTorch.
---
### Import the following libraries
```
import torch
import torchvision.transforms as T
```
### Step1 : Create a class 
For detail explanation refer heliographic_transformation.py 
```
class CustomTransforms:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        
    def __call__(self, image) -> torch.Tensor:
        """
        perform required transformations
        """
        return transformed image
```
### Step2 : Intializing the transforms object
```
custom_transforms =  T.Compose([CustomTransforms, ..., additional transforms functions])
```
### Step3 : Call the transforms 
```
custom_img = custom_transforms(image)
```