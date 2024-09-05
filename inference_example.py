from backbones import SwiftFaceFormer_XXS

import torch
from torchvision.io import read_image
from torchvision.transforms import Resize

img_tensor = read_image("test.jpeg").type(torch.FloatTensor)
batch_img = Resize((112,112))(img_tensor).reshape((1,3,112,112))


#param distillation=False for no dual head distillation
model = SwiftFaceFormer_XXS(distillation=True, num_classes=0)

print(model.load_state_dict(torch.load('models/swiftfaceformer-XXS-KD-MSE-arcface-retina/458592backbone.pth', map_location=torch.device('cpu')), strict=False))

model.eval()

with torch.no_grad():
    print(model(batch_img))
from backbones import SwiftFaceFormer_XXS

import torch
from torchvision.io import read_image
from torchvision.transforms import Resize

img_tensor = read_image("test.jpeg").type(torch.FloatTensor)
batch_img = Resize((112,112))(img_tensor).reshape((1,3,112,112))


#param distillation=False for no dual head distillation
model = SwiftFaceFormer_XXS(distillation=True, num_classes=0)

print(model.load_state_dict(torch.load('models/swiftfaceformer-XXS-KD-MSE-arcface-retina/458592backbone.pth', map_location=torch.device('cpu')), strict=False))

model.eval()

with torch.no_grad():
    print(model(batch_img))