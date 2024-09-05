import sys
#sys.path.append("..")
import backbones

def get_model(name, **kwargs):
    if name=="SwiftFaceFormer_XXS":
        model = backbones.SwiftFaceFormer_XXS(distillation=False, num_classes=0, **kwargs)
    elif name=="SwiftFaceFormer_XS":
        model = backbones.SwiftFaceFormer_XS(distillation=False, num_classes=0, **kwargs)
    elif name =="SwiftFaceFormer_S":
        model = backbones.SwiftFaceFormer_S(distillation=False, num_classes=0, **kwargs)
    elif name == "SwiftFaceFormer_L1":
        model = backbones.SwiftFaceFormer_L1(distillation=False, num_classes=0, **kwargs)
    elif name == "SwiftFaceFormer_L3":
        model = backbones.SwiftFaceFormer_L3(distillation=False, num_classes=0, **kwargs)
    
    return model