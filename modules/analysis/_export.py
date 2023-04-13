from ..slide_info._calculate_property import GetBoundDic
from ..patch._image import ExtractPatchImage, GetWSIThumbnail

from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, roc_auc_score, recall_score, roc_curve, f1_score, auc
from matplotlib.pyplot import suptitle, savefig, subplot, figure, imshow, title, close, plot, axis, clf
from torchvision.transforms import Normalize, ToTensor, Compose, Resize
from numpy.random import choice
from numpy import savetxt, argwhere, argsort, unique, hstack, array, round, uint8, ceil, sum, max, min, eye
from PIL.Image import fromarray
from cv2 import COLORMAP_JET, applyColorMap, resize
from tifffile import imwrite
from os.path import isdir, join
from os import mkdir


def ExportNumericalResults(savePath, resultsDic, resultsSubjects):

    originalClasses = resultsDic["originalClasses"]
    predictionClasses = resultsDic["predictionClasses"]
    probability = resultsDic["probability"]

    resultsInfos = []
    if "Accuracy" in resultsSubjects:
        resultsInfos.append(str("Accuracy : %f\n"%accuracy_score(originalClasses, predictionClasses)))
    if "F1 score" in resultsSubjects:
        resultsInfos.append(str("F1 score : %f\n"%f1_score(originalClasses, predictionClasses, average='macro')))
    if "Precision score" in resultsSubjects:
        resultsInfos.append(str("Precision score : %f\n"%precision_score(originalClasses, predictionClasses, average='macro')))
    if "Sensitivity score" in resultsSubjects:
        resultsInfos.append(str("Sensitivity score : %f\n"%recall_score(originalClasses, predictionClasses, average='macro')))
    if "Classification report" in resultsSubjects:
        resultsInfos.append(str("Classification report : \n" + classification_report(originalClasses, predictionClasses)))

    savetxt(savePath+"numerical_results.txt", array(resultsInfos), fmt="%s", delimiter="\n")

    return None

def ExportROCCurveMultiClass(savePath, resultsDic):

    labelInfosList = array(resultsDic["labelInfos"])
    originalClasses = array(resultsDic["originalClasses"])
    predictionClasses = array(resultsDic["predictionClasses"])
    probability = array(resultsDic["probability"])

    labels = unique(originalClasses)
    labelNum = len(labels)

    for label in labels:
        probabilityPrediction = probability[:,0].copy()
        probabilityPrediction[originalClasses != label] += -1
        probabilityPrediction[originalClasses != label] *= -1

        x, y, _ = roc_curve(eye(labelNum)[originalClasses][:,label], probabilityPrediction, pos_label=1)
        title("label %d AUC %f"%(label, auc(x, y)))
        plot(x,y)
        savefig(savePath+"/ROC_Curve_label_%d.png"%(label))
        close()

    return None

def ExportTopNImages(savePath, ID2Handle, resultsDic, exportNum=20):
    
    IDList = array(resultsDic["ID"])
    patchInfosList = array(resultsDic["patchInfos"], dtype=object)
    labelInfosList = array(resultsDic["labelInfos"])
    originalClasses = array(resultsDic["originalClasses"])
    predictionClasses = array(resultsDic["predictionClasses"])
    probabilityONP = array(resultsDic["probability"])

    labels = unique(originalClasses)
    idxsCorrect = originalClasses == predictionClasses

    for label in labels:
        boolCorrect = originalClasses == predictionClasses
        boolLabel = originalClasses == label
        idxsCorrect = argwhere(boolCorrect * boolLabel).T[0]
        idxTopN = argsort(probabilityONP[:,1][boolCorrect * boolLabel])[-exportNum:]

        figure(figsize=(10, ceil(exportNum/5)*2))
        suptitle("Top %d patch image at label %d"%(exportNum, label))
        for i, idx in enumerate(idxsCorrect[idxTopN]):
            ID = IDList[idx]
            locate, level, size = patchInfosList[idx]
            img = array(ID2Handle[ID]["pathWSI"].read_region(locate, level, size).convert("RGB"))
            subplot(4,5,i+1)
            imshow(img)
            axis("off")
        savefig(savePath+"/Top_%d_patch_image_label%d.png"%(exportNum, label))
        close()

    return None

def ExportClassActivationMapResults(savePath, model, ID2Handle, resultsDic, device, className, imageSize=256, topRank=False, exportNum=20):
    
    model.eval()

    IDList = array(resultsDic["ID"])
    patchInfosList = array(resultsDic["patchInfos"], dtype=object)
    labelInfosList = array(resultsDic["labelInfos"])
    originalClasses = array(resultsDic["originalClasses"])
    predictionClasses = array(resultsDic["predictionClasses"])
    probabilityONP = array(resultsDic["probability"])

    idxList = []
    if topRank:
        labels = unique(originalClasses)
        idxsCorrect = originalClasses == predictionClasses

        for label in labels:
            boolCorrect = originalClasses == predictionClasses
            boolLabel = originalClasses == label
            idxsCorrect = argwhere(boolCorrect * boolLabel).T[0]
            idxTopN = argsort(probabilityONP[:,1][boolCorrect * boolLabel])[-exportNum//len(labels):]
            idxList.extend(idxsCorrect[idxTopN])
    else:
        idxList.extend(choice(range(len(IDList)), exportNum))

    parameters = model.state_dict()
    weightFullyConnected = parameters["fully_connected.weight"].numpy()

    normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )

    dataTransform = Compose([
        Resize((imageSize, imageSize)),
        ToTensor(),
        normalize,
        ])
    
    pathDirCAM = join(savePath, "CAM/")
    if not isdir(pathDirCAM):
        mkdir(pathDirCAM)
    
    for idx in idxList:

        img = array(ID2Handle[IDList[idx]]["pathWSI"].read_region(patchInfosList[idx][0], patchInfosList[idx][1], patchInfosList[idx][2]).convert("RGB"))
        imgSizeW = img.shape[1]
        imgSizeH = img.shape[0]

        FCLs, labels = model(dataTransform(fromarray(img)).unsqueeze(dim=0).to(device))
        FCL = FCLs[0].detach().numpy()
        _sizeC = FCL.shape[0]
        _sizeW = FCL.shape[1]
        _sizeH = FCL.shape[2]
        CAM = weightFullyConnected[originalClasses[idx]].dot(FCL.reshape(_sizeC, _sizeW*_sizeH)).reshape(_sizeW, _sizeH)

        CAM -= min(CAM)
        CAM /= max(CAM)
        CAM *= 255
        CAM = resize(CAM, dsize=(imgSizeW, imgSizeH))
        CAM = applyColorMap(CAM.astype(uint8), COLORMAP_JET)

        figure(figsize=(15, 4 * (img.shape[0] / img.shape[1] + 0.3)))
        suptitle("%s predict %.4f%%"%(labelInfosList[idx], probabilityONP[idx][1]*100))
        subplot(1,3,1)
        imshow(img)
        axis("off")
        subplot(1,3,2)
        imshow((img*0.7+CAM*0.3).astype(uint8))
        axis("off")
        subplot(1,3,3)
        imshow(CAM)
        axis("off")
        savefig(pathDirCAM+"/%3d.png"%idx)
        close()

    return None

def ExportIntersectOverUnionResults(savePath, maskFullDic, ID2Handle, threshold):

    if not isdir(savePath):
        mkdir(savePath)
    if not isdir(join(savePath, "IoU")):
        mkdir(join(savePath, "IoU"))

    recordText = []
    for ID in maskFullDic:

        maskPrediction = maskFullDic[ID]["matrix"].copy()
        shapePrediction = maskPrediction.shape
        maskPrediction[maskPrediction < threshold] = 0
        maskPrediction[maskPrediction >= threshold] = 1

        shapeOriginal = ID2Handle[ID]["pathAnno"].shape
        maskOriginal = ID2Handle[ID]["pathAnno"][0:shapeOriginal[0],0:shapeOriginal[1]]
        maskOriginal = round(resize(maskOriginal, dsize=maskPrediction.shape[::-1]) / 255).astype(uint8)

        maskAddition = maskOriginal + maskPrediction
        intersect = sum(maskAddition == 2)
        union = sum(maskAddition != 0)
        IoU = intersect / union

        OS = ID2Handle[ID]["pathWSI"]
        boundDic = GetBoundDic(OS)
        img = GetWSIThumbnail(OS, 16, boundDic)
        maskOriginal = resize(maskOriginal, dsize=(img.shape[1], img.shape[0]))
        maskPrediction = resize(maskPrediction, dsize=(img.shape[1], img.shape[0]))
        figure(figsize=(21, 7 * (img.shape[0] / img.shape[1]) + 0.5))
        suptitle("%s, IoU : %.2f%%"%(ID, IoU*100))
        subplot(1,3,1)
        title("Annotation")
        _img = img.copy()
        _img[maskOriginal == 0] = (img[maskOriginal == 0] / 3 * 2).astype(uint8)
        imshow(_img)
        axis("off")
        subplot(1,3,2)
        title("WSI")
        imshow(img)
        axis("off")
        subplot(1,3,3)
        title("Diagnosis")
        _img = img.copy()
        _img[maskPrediction == 0] = (img[maskPrediction == 0] / 3 * 2).astype(uint8)
        imshow(_img)
        axis("off")
        savefig(join(savePath, "IoU", "%s_IoU.png"%(ID)))
        close()
        recordText.append([ID, IoU])

    savetxt(join(savePath, "IoU_list.tsv"), array(recordText), delimiter="\t", fmt="%s")

    return None

def ExportROCCurveMatrix(savePath, maskFullDic, ID2Handle):

    if not isdir(savePath):
        mkdir(savePath)
    if not isdir(join(savePath, "ROC")):
        mkdir(join(savePath, "ROC"))

    originalClasses = []
    probabilityPrediction = []

    recordText = []
    for ID in maskFullDic:

        maskPrediction = maskFullDic[ID]["matrix"] / 255

        shapeOriginal = ID2Handle[ID]["pathAnno"].shape
        maskOriginal = ID2Handle[ID]["pathAnno"][0:shapeOriginal[0],0:shapeOriginal[1]]
        maskOriginal = round(resize(maskOriginal, dsize=maskPrediction.shape[::-1]) / 255).astype(uint8)

        x, y, _ = roc_curve(eye(2)[hstack(maskOriginal)][:,1], hstack(maskPrediction), pos_label=1)
        _auc = auc(x, y)
        title("%s, AUC %.2f%%"%(ID, _auc*100))
        plot(x,y)
        savefig(join(savePath, "ROC", "%s_ROC_Curve.png"%(ID)))
        close()

        originalClasses.extend(hstack(maskOriginal).tolist())
        probabilityPrediction.extend(hstack(maskPrediction).tolist())
        recordText.append([ID, _auc])

    originalClasses = array(originalClasses)
    probabilityPrediction = array(probabilityPrediction)

    x, y, _ = roc_curve(eye(2)[originalClasses][:,1], probabilityPrediction, pos_label=1)
    title("Total AUC %f"%(auc(x, y)))
    plot(x,y)
    savefig(join(savePath, "ROC", "Total_ROC_Curve.png"))
    close()

    savetxt(join(savePath, "ROC_list.tsv"), array(recordText), delimiter="\t", fmt="%s")

    return None