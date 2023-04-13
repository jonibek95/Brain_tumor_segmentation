from ..slide_info._calculate_property import GetMicronPerPixelDic, GetBoundDic
from ._calculate import ConvertMicron2Pixel

from openslide import OpenSlide
from pytiff import Tiff
from numpy.random import permutation
from numpy import meshgrid, arange, argmin, round, array, vstack, int32, zeros, mean, sum
from cv2 import imread


def CalculateAptLevel(levelDownsamples, sizeSepXY, sizeDataXY):

    sizeSepXYDownsamples = round(sizeSepXY / vstack(levelDownsamples)).astype(int32)
    sizeSepXYDownsamplesDiff = sum(sizeSepXY / vstack(levelDownsamples) - sizeDataXY, axis=1)
    sizeSepXYDownsamplesDiff[sizeSepXYDownsamplesDiff < 0] *= -2
    level = argmin(sizeSepXYDownsamplesDiff)
    sizePixelXY = sizeSepXYDownsamples[level]

    return sizePixelXY, level

def CalculateCoordinateArray(sizeSepWH, sizeIntervalWH, boundDic, randomShuffle):
    
    coordinateX = arange(boundDic['x'], boundDic['w'] - sizeSepWH[0], sizeIntervalWH[0])
    coordinateY = arange(boundDic['y'], boundDic['h'] - sizeSepWH[1], sizeIntervalWH[1])
    coordinateXY = array(meshgrid(coordinateX, coordinateY)).T.reshape(-1,2)
    if randomShuffle:
        coordinateXY = permutation(coordinateXY)

    return coordinateXY

def CalculateMaskSeparation(maskHandle, boundDic, coordinateXY, sizeSepWH):

    ratioW = boundDic['w'] / maskHandle.shape[1]
    ratioH = boundDic['h'] / maskHandle.shape[0]
    ratioWH = array([ratioW, ratioH])
    initXY = array([boundDic['x'], boundDic['y']])
    coordinateXY = round((coordinateXY - initXY) / ratioWH).astype(int32)
    sizeSepXY = round(sizeSepWH / ratioWH).astype(int32)

    return coordinateXY, sizeSepXY

def GetMaskAnnotationInfos(idx, argsDic):

    handleMask = argsDic["handleMask"]
    x, y = argsDic["coordinateXY"][idx]
    w, h = argsDic["sizeSepWH"]

    if mean(handleMask[y:y+h,x:x+w]) != 0:
        label = "positive"
    else:
        label = "negative"

    labelInfos = {}
    labelInfos["location"] = (x, y)
    labelInfos["size"] = (w, h)
    labelInfos["label"] = label
    labelInfos["labels"] = ["positive", "negative"]

    return label, labelInfos

def GetMaskLabelInfos(idx, argsDic):

    label = argsDic["label"]
    labels = argsDic["labels"]

    labelInfos = {}
    labelInfos["label"] = label
    labelInfos["labels"] = labels

    return label, labelInfos

def GetMasUnknownInfos(idx, argsDic):

    label = "Unknown"
    labelInfos = [label]

    return label, labelInfos

def MakePatchInfosDicList(ID, level, boundDic, coordinateXY, sizePixelWH, handleDiag, coordinateXYDiag, sizeSepWHDiag, allowRatio, LabelFunc, maskFuncArgsDic):

    patchInfosDicPack = {}
    patchInfosDicList = []
    for idx in range(len(coordinateXY)):
        x, y = coordinateXY[idx]
        w, h = sizePixelWH
        xDiag, yDiag = coordinateXYDiag[idx]
        wDiag, hDiag = sizeSepWHDiag
        maskDiagSep = handleDiag[yDiag:yDiag+hDiag,xDiag:xDiag+wDiag]
        diagMean = mean(maskDiagSep / 255)
        if diagMean > allowRatio:
            patchInfosDic = {}
            patchInfosDic["ID"] = ID
            patchInfosDic["patchInfos"] = {}
            patchInfosDic["patchInfos"]["location"] = (x, y)
            patchInfosDic["patchInfos"]["level"] = level
            patchInfosDic["patchInfos"]["size"] = (w, h)
            label, patchInfosDic["labelInfos"] = LabelFunc(idx, maskFuncArgsDic)
            if label not in patchInfosDicPack:
                patchInfosDicPack[label] = []
            patchInfosDicPack[label].append(patchInfosDic)

    return patchInfosDicPack

def AnnotatedDiagPatchInfos(ID, WSIDic, sizeDataWH, sizeMicronWH, intervalMicronWH, randomShuffle=True, allowRatio=0.1):

    pathWSI = WSIDic["pathWSI"]
    pathDiag = WSIDic["pathDiag"]
    pathAnno = WSIDic["pathAnno"]

    OS = OpenSlide(pathWSI)
    boundDic = GetBoundDic(OS)
    mppDic = GetMicronPerPixelDic(OS)
    handleDiag = Tiff(pathDiag)
    handleAnno = Tiff(pathAnno)
    ID2Path = {}
    ID2Path["pathWSI"] = pathWSI
    ID2Path["pathAnno"] = pathAnno

    sizeSepWH = array(ConvertMicron2Pixel(sizeMicronWH, mppDic))
    sizeIntervalWH = array(ConvertMicron2Pixel(intervalMicronWH, mppDic))

    coordinateXY = CalculateCoordinateArray(sizeSepWH, sizeIntervalWH, boundDic, randomShuffle)

    sizePixelWH, level = CalculateAptLevel(OS.level_downsamples, sizeSepWH, sizeDataWH)

    coordinateXYDiag, sizeSepWHDiag = CalculateMaskSeparation(handleDiag, boundDic, coordinateXY, sizeSepWH)
    coordinateXYAnno, sizeSepWHAnno = CalculateMaskSeparation(handleAnno, boundDic, coordinateXY, sizeSepWH)

    maskFuncArgsDic = {"handleMask" : handleAnno, "coordinateXY" : coordinateXYAnno, "sizeSepWH" : sizeSepWHAnno}
    patchInfosDicPack = MakePatchInfosDicList(ID, level, boundDic, coordinateXY, sizePixelWH,
                                              handleDiag, coordinateXYDiag, sizeSepWHDiag,
                                              allowRatio, GetMaskAnnotationInfos, maskFuncArgsDic)

    patchNum = 0
    for key in patchInfosDicPack:
        patchNum += len(patchInfosDicPack[key])
    print(ID, patchNum)

    return ID, ID2Path, patchInfosDicPack

def AnnotatedDiagPatchInfosMultiProcessing(inputDic):

    ID = inputDic["ID"]
    WSIDic = inputDic["WSIDic"]
    sizeDataWH = inputDic["sizeDataWH"]
    sizeMicronWH = inputDic["sizeMicronWH"]
    intervalMicronWH = inputDic["intervalMicronWH"]
    
    return AnnotatedDiagPatchInfos(ID, WSIDic, sizeDataWH, intervalMicronWH, sizeMicronWH)

def LabeledDiagPatchInfos(ID, WSIDic, labels, sizeDataWH, sizeMicronWH, intervalMicronWH, randomShuffle=True, allowRatio=0.7):

    pathWSI = WSIDic["pathWSI"]
    pathDiag = WSIDic["pathDiag"]
    label = WSIDic["label"]

    OS = OpenSlide(pathWSI)
    boundDic = GetBoundDic(OS)
    mppDic = GetMicronPerPixelDic(OS)
    handleDiag = Tiff(pathDiag)
    ID2Path = {}
    ID2Path["pathWSI"] = pathWSI

    sizeSepWH = array(ConvertMicron2Pixel(sizeMicronWH, mppDic))
    sizeIntervalWH = array(ConvertMicron2Pixel(intervalMicronWH, mppDic))

    coordinateXY = CalculateCoordinateArray(sizeSepWH, sizeIntervalWH, boundDic, randomShuffle)

    sizePixelWH, level = CalculateAptLevel(OS.level_downsamples, sizeSepWH, sizeDataWH)

    coordinateXYDiag, sizeSepWHDiag = CalculateMaskSeparation(handleDiag, boundDic, coordinateXY, sizeSepWH)

    maskFuncArgsDic = {"label" : label, "labels" : labels}
    patchInfosDicPack = MakePatchInfosDicList(ID, level, boundDic, coordinateXY, sizePixelWH,
                                              handleDiag, coordinateXYDiag, sizeSepWHDiag,
                                              allowRatio, GetMaskLabelInfos, maskFuncArgsDic)

    patchNum = 0
    for key in patchInfosDicPack:
        patchNum += len(patchInfosDicPack[key])
    print(ID, patchNum)

    return ID, ID2Path, patchInfosDicPack

def LabeledDiagPatchInfosMultiProcessing(inputDic):

    ID = inputDic["ID"]
    WSIDic = inputDic["WSIDic"]
    labels = inputDic["labels"]
    sizeDataWH = inputDic["sizeDataWH"]
    sizeMicronWH = inputDic["sizeMicronWH"]
    intervalMicronWH = inputDic["intervalMicronWH"]
    
    return LabeledDiagPatchInfos(ID, WSIDic, labels, sizeDataWH, intervalMicronWH, sizeMicronWH)


def UnknownDiagPatchInfos(ID, WSIDic, sizeDataWH, sizeMicronWH, intervalMicronWH, randomShuffle=True, allowRatio=0.7):

    pathWSI = WSIDic["pathWSI"]
    pathDiag = WSIDic["pathDiag"]

    OS = OpenSlide(pathWSI)
    boundDic = GetBoundDic(OS)
    mppDic = GetMicronPerPixelDic(OS)
    handleDiag = Tiff(pathDiag)
    ID2Path = {}
    ID2Path["pathWSI"] = pathWSI

    sizeSepWH = array(ConvertMicron2Pixel(sizeMicronWH, mppDic))
    sizeIntervalWH = array(ConvertMicron2Pixel(intervalMicronWH, mppDic))

    coordinateXY = CalculateCoordinateArray(sizeSepWH, sizeIntervalWH, boundDic, randomShuffle)

    sizePixelWH, level = CalculateAptLevel(OS.level_downsamples, sizeSepWH, sizeDataWH)

    coordinateXYDiag, sizeSepWHDiag = CalculateMaskSeparation(handleDiag, boundDic, coordinateXY, sizeSepWH)

    maskFuncArgsDic = {}
    patchInfosDicPack = MakePatchInfosDicList(ID, level, boundDic, coordinateXY, sizePixelWH,
                                              handleDiag, coordinateXYDiag, sizeSepWHDiag,
                                              allowRatio, GetMasUnknownInfos, maskFuncArgsDic)

    patchNum = 0
    for key in patchInfosDicPack:
        patchNum += len(patchInfosDicPack[key])
    print(ID, patchNum)

    return ID, ID2Path, patchInfosDicPack

def UnknownDiagPatchInfosMultiProcessing(inputDic):

    ID = inputDic["ID"]
    WSIDic = inputDic["WSIDic"]
    sizeDataWH = inputDic["sizeDataWH"]
    sizeMicronWH = inputDic["sizeMicronWH"]
    intervalMicronWH = inputDic["intervalMicronWH"]
    
    return UnknownDiagPatchInfos(ID, WSIDic, sizeDataWH, sizeMicronWH, intervalMicronWH)
