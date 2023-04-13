from ._utils import *
from ._annotation import *

def MakeDiagnosisMask(thumbImage, infoDic):
    
    imgDic = SplitLHSV(thumbImage)
    maskSelect = SelectRange(imgDic, infoDic["RangeList"])
    maskStrong = GaussianBlurNThreasholding(maskSelect, infoDic["GauStrong"])
    maskWeak = GaussianBlurNThreasholding(maskSelect, infoDic["GauWeak"])
    maskStrongRegion = SelectRectRegion(maskStrong)
    mask = maskWeak * maskStrongRegion
    
    return mask
