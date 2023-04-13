from .res_U_net import *
from .dsmil import *


def ResUNext(layers=[3, 4, 6, 3, 6, 4, 3], num_classes=1000, groups=32, width_per_group=8, export_FCL=False):
    return ResUNetClassification(layers, num_classes=num_classes, groups=32, width_per_group=8, export_FCL=export_FCL)

def ResUNetSeg(layers=[3, 4, 6, 3, 6, 4, 3], groups=32, width_per_group=8):
    return ResUNetSegmentation(layers, groups=32, width_per_group=8)