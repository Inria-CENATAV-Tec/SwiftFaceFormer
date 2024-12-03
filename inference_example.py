from backbones import SwiftFaceFormer_XXS

import torch
from torchvision.io import read_image
from torchvision.transforms import Resize

img_tensor = read_image("test.jpg").type(torch.FloatTensor) #assume image is aligned to insightface template
img_tensor = ((img_tensor/255.0) - 0.5) / 0.5
batch_img = Resize((112,112))(img_tensor).reshape((1,3,112,112))


#param distillation=False for no dual head distillation
model = SwiftFaceFormer_XXS(distillation=True, num_classes=0)

checkpoint = 'models/KD-SFF-L3-XXS/SwiftFaceFormer-XXS-MSE-arcface-retina/458592backbone.pth'
print(model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')), strict=False))

model.eval()

with torch.no_grad():
    embedding = model(batch_img)

print(embedding)
print(embedding.shape)
