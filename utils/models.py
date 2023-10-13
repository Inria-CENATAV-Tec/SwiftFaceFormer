import sys
#sys.path.append("..")
import backbones

def get_model(name, **kwargs):
    if name=="SwiftFormer_XS":
        model = backbones.SwiftFormer_XS(distillation=False, num_classes=0, **kwargs)
    elif name =="SwiftFormer_S":
        model = backbones.SwiftFormer_S(distillation=False, num_classes=0, **kwargs)
    elif name == "SwiftFormer_L1":
        model = backbones.SwiftFormer_L1(distillation=False, num_classes=0, **kwargs)
    elif name == "SwiftFormer_L3":
        model = backbones.SwiftFormer_L3(distillation=False, num_classes=0, **kwargs)
    
    return model