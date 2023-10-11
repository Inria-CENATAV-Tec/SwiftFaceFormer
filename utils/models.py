import sys
sys.path.append("..")
import backbones

def get_model(name):
    if name=="SwiftFormer_XS":
        backbones.SwiftFormer_XS()