from ..slide_info._calculate_property import *
from ..patch._calculate import *

from os.path import isfile, isdir
from os import listdir, mkdir
from pickle import dump, load
from numpy import savetxt, float32, where, uint8, array, round, zeros, ceil, load, mean


def RecordCSVAnnotation(variableFileName, processRecord, inputValue):

    recordLoss = 1e3
    recordEpoch = 1000

    for epoch in processRecord["loss"].keys():
        if recordLoss > processRecord["loss"][epoch]["valid"]:
            recordEpoch = epoch
            recordLoss = processRecord["loss"][epoch]["valid"]

    modelPath = "results/%s/loss_%.4f_epoch_%d.pt"%(variableFileName, recordLoss, recordEpoch)
    dataInfosPath = "results/%s/valid_loss_%.4f_epoch_%d.dump"%(variableFileName, recordLoss, recordEpoch)
    matrixPackPath = "results/%s/whole_mask_loss_%.4f_epoch_%d.dump"%(variableFileName, recordLoss, recordEpoch)

    saveCSV = array([["modelPath", modelPath],
                     ["dataInfosPath", dataInfosPath],
                     ["matrixPackPath", matrixPackPath],
                     ["ID2PathTrain", inputValue["ID2PathTrain"]],
                     ["patchInfosTrain", inputValue["patchInfosTrain"]],
                     ["ID2PathValid", inputValue["ID2PathValid"]],
                     ["patchInfosValid", inputValue["patchInfosValid"]],
                     ["imageSize", inputValue["imageSize"]]])
    
    savetxt("%s_analysis_recommended.csv"%(variableFileName), saveCSV, delimiter=',', fmt="%s")

    return None

def RecordCSVLabel(variableFileName, processRecord, inputValue, labels):

    recordLoss = 1e3
    recordAccuracy = 1e3
    recordEpoch = 1000

    for epoch in processRecord["loss"].keys():
        if recordLoss > processRecord["loss"][epoch]["valid"]:
            recordEpoch = epoch
            recordLoss = processRecord["loss"][epoch]["valid"]
            recordAccuracy = processRecord["accuracy"][epoch]["valid"]

    modelPath = "results/%s/loss_%.4f_acc_%.2f%%_epoch_%d.pt"%(variableFileName, recordLoss, recordAccuracy, recordEpoch)
    dataInfosPath = "results/%s/valid_loss_%.4f_acc_%.2f%%_epoch_%d.dump"%(variableFileName, recordLoss, recordAccuracy, recordEpoch)
    resultsSubjects = "Accuracy/F1 score/Precision score/Sensitivity score/Classification report"
    labelString = ""
    for label in labels:
        labelString += label
        labelString += ':'

    saveCSV = array([["modelPath", modelPath],
                     ["dataInfosPath", dataInfosPath],
                     ["ID2PathTrain", inputValue["ID2PathTrain"]],
                     ["patchInfosTrain", inputValue["patchInfosTrain"]],
                     ["ID2PathValid", inputValue["ID2PathValid"]],
                     ["patchInfosValid", inputValue["patchInfosValid"]],
                     ["labels", labelString],
                     ["imageSize", inputValue["imageSize"]],
                     ["resultsSubjects", resultsSubjects]])
    
    savetxt("%s_analysis_recommended.csv"%(variableFileName), saveCSV, delimiter=',', fmt="%s")

    return None

def SaveMaskFullDic(rootPath, ID2Handle, pathSave, intervalMicronSize=200):

    intervalMicronWH = (intervalMicronSize, intervalMicronSize)
    labelInfos = {}
    maskFullDic = {}
    for name in listdir(rootPath):
        mask = load(rootPath+name)

        nameSplit = name[:-4].split(':')
        ID = nameSplit[0]
        x, y = nameSplit[1].split(',')
        x, y = int(x), int(y)
        w, h = nameSplit[2].split(',')
        w, h = int(w), int(h)

        if ID not in maskFullDic:
            boundDic = GetBoundDic(ID2Handle[ID]["pathWSI"])
            mppDic = GetMicronPerPixelDic(ID2Handle[ID]["pathWSI"])
            sizeIntervalWH = ConvertMicron2Pixel(intervalMicronWH, mppDic)

            if "pathAnno" in ID2Handle[ID]:
                wRatio = mask.shape[1] / w
                hRatio = mask.shape[0] / h
                wFull = ID2Handle[ID]["pathAnno"].shape[1]
                hFull = ID2Handle[ID]["pathAnno"].shape[0]
            else:
                wRatio = sizeIntervalWH[0] / w
                hRatio = sizeIntervalWH[1] / h
                wFull = boundDic['w'] / sizeIntervalWH[0]
                hFull = boundDic['h'] / sizeIntervalWH[1]
                mask = mean(mask)
            maskFullDic[ID] = {}
            maskFullDic[ID]["bound"] = boundDic
            maskFullDic[ID]["ratio"] = (wRatio, hRatio)
            maskFullDic[ID]["matrix"] = zeros(ceil((hFull * hRatio, wFull * wRatio)).astype(int), dtype=float32)
            maskFullDic[ID]["matrixCount"] = zeros(ceil((hFull * hRatio, wFull * wRatio)).astype(int), dtype=float32)
        else:
            wRatio, hRatio = maskFullDic[ID]["ratio"]

        xMask = int(round((x - maskFullDic[ID]["bound"]['x']) * wRatio))
        yMask = int(round((y - maskFullDic[ID]["bound"]['y']) * hRatio))
        wMask = int(round(w * wRatio))
        hMask = int(round(h * hRatio))
        print(w * wRatio, h * hRatio)
        maskFullDic[ID]["matrix"][yMask:yMask+hMask,xMask:xMask+wMask] += mask
        maskFullDic[ID]["matrixCount"][yMask:yMask+hMask,xMask:xMask+wMask] += 1

    for ID in maskFullDic:
        indexs = where(maskFullDic[ID]["matrixCount"] != 0)
        maskFullDic[ID]["matrix"][indexs] = (maskFullDic[ID]["matrix"][indexs] / maskFullDic[ID]["matrixCount"][indexs] * 255)
        del maskFullDic[ID]["matrixCount"]
        maskFullDic[ID]["matrix"] = maskFullDic[ID]["matrix"].astype(uint8)

    dump(maskFullDic, open(pathSave, "wb"))

    return None

def VariableDumpSaveNLoad(key, func, *args):

    pathDump = "dump_dir/"
    if not isdir(pathDump):
        mkdir(pathDump)

    fileName = pathDump+key+".dump"
    if isfile(fileName):
        value = load(open(fileName, "rb"))
    else:
        value = func(*args)
        dump(value, open(fileName, "wb"))
    key = value

    return key