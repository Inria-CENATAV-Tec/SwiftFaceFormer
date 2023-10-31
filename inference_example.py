from backbones import SwiftFormer_XXS

import torch
from torchvision.io import read_image
from torchvision.transforms import Resize

img_tensor = read_image("./1547482358250.jpeg").type(torch.FloatTensor)
batch_img = Resize((112,112))(img_tensor).reshape((1,3,112,112))


#param distillation=False for no dual head distillation
model = SwiftFormer_XXS(distillation=True, num_classes=0)

print(model.load_state_dict(torch.load('models/swiftformer-XXS-KD-MSE-arcface-retina/458592backbone.pth', map_location=torch.device('cpu')), strict=False))

model.eval()

with torch.no_grad():
    print(model(batch_img))