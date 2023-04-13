from numpy import where, zeros, ones, max, min
from cv2 import COLOR_BGR2HSV_FULL, COLOR_BGR2GRAY, CV_8U, GaussianBlur, Laplacian, cvtColor, inRange, split


def SplitLHSV(image):
    L = Laplacian(cvtColor(image, COLOR_BGR2GRAY), CV_8U, ksize=3)
    H, S, V = split(cvtColor(image, COLOR_BGR2HSV_FULL))
    return {'L' : L, 'H' : H, 'S' : S, 'V' : V}

def SelectRange(imgDic, RangeList):

    maskSum = zeros(imgDic['L'].shape)
    for RangeDic in RangeList:
        maskSumSub = ones(imgDic['L'].shape)
        for imgType in RangeDic:
            maskSumSub *= inRange(imgDic[imgType], RangeDic[imgType][0], RangeDic[imgType][1])
        maskSum += maskSumSub

    maskSum[maskSum > 0] = 1

    return maskSum

def GaussianBlurNThreasholding(image, GauThrDic):
    image = GaussianBlur(image, (GauThrDic['X'], GauThrDic['X']), GauThrDic["Sigma"])
    image = image > GauThrDic["Threshold"]
    return image

def SelectRectRegion(mask):

    xMin = min(where(mask)[0])
    xMax = max(where(mask)[0])
    yMin = min(where(mask)[1])
    yMax = max(where(mask)[1])

    padX = int((xMax - xMin)*0.05)
    padY = int((yMax - yMin)*0.05)

    xMin = xMin - padX if xMin - padX > 0 else 0
    xMax = xMax + padX if xMax + padX < mask.shape[0] else mask.shape[0]
    yMin = yMin - padY if yMin - padY > 0 else 0
    yMax = yMax + padY if yMax + padY < mask.shape[1] else mask.shape[1]

    mask[:,:] = False
    mask[xMin:xMax,yMin:yMax] = True

    return mask
