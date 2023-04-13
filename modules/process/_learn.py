from ..analysis._numerical import ConvertLabelNOutput2Softmax

from torch.autograd import Variable
from torch.cuda import current_device
from torch import no_grad, max, min
from matplotlib.pyplot import savefig, subplot, legend, figure, title, xlabel, ylabel, plot, clf
from numpy import save
from cv2 import ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, ROTATE_180, rotate, resize, flip
from os.path import isdir
from os import mkdir


def DeformMatrix(matrix, deformKey):

    infoFlipW = deformKey["flipW"]
    infoFlipH = deformKey["flipH"]
    infoAngleD = deformKey["rotation"]

    if infoFlipH:
        matrix = flip(matrix, 0)
    if infoFlipW:
        matrix = flip(matrix, 1)

    if infoAngleD > 180:
        infoAngleD -= 360
    elif infoAngleD <= -180:
        infoAngleD += 360

    if infoAngleD == 90:
        matrix = rotate(matrix, ROTATE_90_CLOCKWISE)
    elif infoAngleD == 180:
        matrix = rotate(matrix, ROTATE_180)
    elif infoAngleD == -90:
        matrix = rotate(matrix, ROTATE_90_COUNTERCLOCKWISE)

    return matrix

def OrganizePatchImageResultsAnnotation(infoDic, dataInfosDic, outputs, status):

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
        if key == "labelInfos":
            location = dataInfosDic["labelInfos"]["location"]
            size = dataInfosDic["labelInfos"]["size"]
            locationZip = zip(location[0].tolist(), location[1].tolist())
            sizeZip = zip(size[0].tolist(), size[1].tolist())
            infoDic[key].extend(list(zip(locationZip, sizeZip)))
            
            locationZip = zip(location[0].tolist(), location[1].tolist())
            sizeZip = zip(size[0].tolist(), size[1].tolist())
            for ID, (x, y), (w, h), output in zip(dataInfosDic["ID"], locationZip, sizeZip, outputs):
                keyName = "%s:%d,%d:%d,%d"%(ID, x, y, w, h)
                if not isdir("temp/"):
                    mkdir("temp/")
                if not isdir("temp/"+status+'/'):
                    mkdir("temp/"+status+'/')
                save("temp/"+status+'/'+keyName+".npy", output)

    return infoDic

def OrganizePatchImageResultsLabel(infoDic, dataInfosDic, labels, outputs, labelClasses, predClasses):

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
        if key == "labelInfos":
            if "label" in dataInfosDic["labelInfos"]:
                label = dataInfosDic["labelInfos"]["label"]
                infoDic[key].extend(label)
            else:
                location = dataInfosDic["labelInfos"]["location"]
                size = dataInfosDic["labelInfos"]["size"]
                locationZip = zip(location[0].tolist(), location[1].tolist())
                sizeZip = zip(size[0].tolist(), size[1].tolist())
                infoDic[key].extend(list(zip(locationZip, sizeZip)))
        if key == "outputMatrix":
            infoDic[key].extend(outputs.tolist())
        if key == "originalClasses":
            infoDic[key].extend(labelClasses.tolist())
        if key == "predictionClasses":
            infoDic[key].extend(predClasses.tolist())
        if key == "probability":
            probability = ConvertLabelNOutput2Softmax(outputs, labelClasses, predClasses)
            infoDic[key].extend(probability)

    return infoDic

def PatchImageDeeplearningProcessingLabeled(model, device, dataLoader, criterion, optimizer, epoch, training=False, announceBatchStep=100):

    if training:
        model.train()
        status = "Train"
    else:
        model.eval()
        status = "Valid"

    correct, current = 0, 0
    accSum, lossSum = 0, 0
    total = len(dataLoader.dataset)
    infos = GetDefaultInfoLabel()

    for _idx, (dataInfosDic, datas, labels) in enumerate(dataLoader):

        datas, labels = datas.to(device), labels.to(device)

        if training:
            datas, labels = Variable(datas), Variable(labels)
            optimizer.zero_grad()
        else:
            with no_grad():
                datas = datas
                labels = labels

        outputs = model(datas)

        loss = criterion(outputs, max(labels, 1)[1])
        if training:
            loss.backward()
            optimizer.step()
        lossSum += loss.cpu().item()

        labelClasses = max(labels, 1)[1].detach().cpu().numpy()
        predClasses = max(outputs, 1)[1].detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        infos = OrganizePatchImageResultsLabel(infos, dataInfosDic, labels, outputs, labelClasses, predClasses)

        correct += (predClasses == labelClasses).sum()
        current += dataLoader.batch_size

        if _idx%announceBatchStep == 0:
            print("%s Epoch %d | [%d/%d (%.2f%%)] Loss %.4f, Accuaracy %.2f%%"
                  %(status, epoch, current, total, (100*current/total), lossSum/current, 100*correct/current),
                  end='\r')

    del dataInfosDic, datas, labels
    current_device()

    lossAvg = lossSum / total
    accAvg = 100 * correct / total
    print("%s Epoch %d | Avarage Loss %.4f, Average Accuracy %.2f%%"%(status, epoch, lossAvg, accAvg))

    return lossAvg, accAvg, infos

def PatchImageDeeplearningProcessingAnnotated(model, device, dataLoader, criterion, optimizer, epoch, training=False, announceBatchStep=100):

    if training:
        model.train()
        status = "Train"
    else:
        model.eval()
        status = "Valid"

    current = 0
    lossSum = 0
    total = len(dataLoader.dataset)
    infos = GetDefaultInfoAnnotation()

    for _idx, (dataInfosDic, datas, labels, deformKeys) in enumerate(dataLoader):

        datas, labels = datas.to(device), labels.to(device)

        if training:
            datas, labels = Variable(datas), Variable(labels)
            optimizer.zero_grad()
        else:
            with no_grad():
                datas = datas
                labels = labels

        outputs = model(datas)

        loss = criterion(outputs, labels.unsqueeze(1))
        if training:
            loss.backward()
            optimizer.step()
        lossSum += loss.cpu().item()

        outputs = outputs.detach().cpu().numpy()
        outputsNew = []
        for output, flipW, flipH, rotation in zip(outputs, deformKeys["flipW"], deformKeys["flipH"], deformKeys["rotation"]):
            rotation *= -1
            deformKey = {}
            deformKey["flipW"] = flipW.item()
            deformKey["flipH"] = flipH.item()
            deformKey["rotation"] = rotation.item()
            outputsNew.append(DeformMatrix(output[0], deformKey))
        outputs = outputsNew

        infos = OrganizePatchImageResultsAnnotation(infos, dataInfosDic, outputs, status)

        current += dataLoader.batch_size

        if _idx%announceBatchStep == 0:
            print("%s Epoch %d | [%d/%d (%.2f%%)] Loss %.4f"
                  %(status, epoch, current, total, (100*current/total), lossSum/current),
                  end='\r')

    del dataInfosDic, datas, labels, deformKeys
    current_device()

    lossAvg = lossSum / total
    print("%s Epoch %d | Avarage Loss %.4f"%(status, epoch, lossAvg))

    return lossAvg, 0, infos

def PlotProcess(processRecord, variableFileName):

    epochs = []
    trainLoss = []
    validLoss = []
    trainAccuracy = []
    validAccuracy = []

    for epoch in processRecord["loss"]:

        epochs.append(epoch)
        trainLoss.append(processRecord["loss"][epoch]["train"])
        validLoss.append(processRecord["loss"][epoch]["valid"])
        trainAccuracy.append(processRecord["accuracy"][epoch]["train"])
        validAccuracy.append(processRecord["accuracy"][epoch]["valid"])

    figure(figsize=(15, 5))
    subplot(1,2,1)
    title("loss")
    plot(epochs, trainLoss, label="train", color="red")
    plot(epochs, validLoss, label="valid", color="blue")
    xlabel("epoch")
    ylabel("loss")
    legend()
    subplot(1,2,2)
    title("accuracy")
    plot(epochs, trainAccuracy, label="train", color="red")
    plot(epochs, validAccuracy, label="valid", color="blue")
    xlabel("epoch")
    ylabel("accuracy")
    legend()
    savefig("%s_PlotProcess.png"%variableFileName)
    clf()

    return None

def GetDefaultInfoAnnotation():
    
    infos = {"ID" : [],
            "patchInfos" : [],
            "labelInfos" : []
            }

    return infos

def GetDefaultInfoLabel():
    
    infos = {"ID" : [],
            "patchInfos" : [],
            "labelInfos" : [],
            "outputMatrix" : [],
            "originalClasses" : [],
            "predictionClasses" : [],
            "probability" : []
            }

    return infos