from torch.autograd import Variable
from torch.cuda import current_device
from torch import no_grad, max, min
from matplotlib.pyplot import savefig, subplot, legend, figure, title, xlabel, ylabel, plot, clf
from numpy import ones, save, exp
from cv2 import resize
from os.path import isdir
from os import mkdir


def OrganizePatchImageResultsLabel(infoDic, dataInfosDic, outputs, predClasses):

    status = "Test"
    for key in infoDic:
        if key == "ID":
            IDList = dataInfosDic["ID"]
            infoDic[key].extend(IDList)
        if key == "patchInfos":
            location = dataInfosDic["patchInfos"]["location"]
            level = dataInfosDic["patchInfos"]["level"]
            size = dataInfosDic["patchInfos"]["size"]
            locationZip = zip(location[0].tolist(), location[1].tolist())
            levelList = level.tolist()
            sizeZip = zip(size[0].tolist(), size[1].tolist())
            infoDic[key].extend(list(zip(locationZip, levelList, sizeZip)))
        if key == "outputMatrix":
            infoDic[key].extend(outputs.tolist())
        if key == "predictionClasses":
            infoDic[key].extend(predClasses.tolist())

    return infoDic

def PatchImageDeeplearningDiagnosisLabeled(model, device, dataLoader, announceBatchStep=100):

    model.eval()

    current = 0
    total = len(dataLoader.dataset)
    infos = GetDefaultInfoLabel()

    for _idx, (dataInfosDic, datas) in enumerate(dataLoader):

        datas = datas.to(device)

        with no_grad():
            datas = datas

        outputs = model(datas)

        predClasses = max(outputs, 1)[1].detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        infos = OrganizePatchImageResultsLabel(infos, dataInfosDic, outputs, predClasses)

        current += dataLoader.batch_size

        if _idx%announceBatchStep == 0:
            print("Diagnosis | [%d/%d (%.2f%%)]"%(current, total, (100*current/total)), end='\r')

    del dataInfosDic, datas
    current_device()

    print("Diagnosis over")

    return infos

def GetDefaultInfoAnnotation():
    
    infos = {"ID" : [],
            "patchInfos" : [],
            "labelInfos" : []
            }

    return infos

def GetDefaultInfoLabel():
    
    infos = {"ID" : [],
            "patchInfos" : [],
            "outputMatrix" : [],
            "predictionClasses" : []
            }

    return infos