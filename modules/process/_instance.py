from torch.autograd import Variable
from torch.cuda import current_device
from torch import no_grad, Tensor, max
from numpy.random import permutation
from numpy import zeros


def GetDefaultInfo():
    
    infos = {"ID" : [],
            "patchInfos" : [],
            "label" : [],
            "labelInfos" : [],
            "outputMatrix" : [],
            }

    return infos

def MultipleInstancelearningProcessing(modelMIL, device, instanceBagDic, criterion, optimizer, epoch, training=False):

    if training:
        modelMIL.train()
        status = "Train"
    else:
        modelMIL.eval()
        status = "Valid"

    current = 0
    correct = 0
    lossSum = 0

    for ID in permutation(list(instanceBagDic.keys())):
        for label in instanceBagDic[ID].keys():

            current += 1

            datas = Tensor(instanceBagDic[ID][label]["instanceBag"]).to(device)

            labels = list(instanceBagDic[ID].keys())
            targets = zeros(len(labels))
            targetsIdx = labels.index(label)
            targets[targetsIdx] = 1
            targets = Tensor(targets).unsqueeze(0).to(device)

            if training:
                datas, targets = Variable(datas), Variable(targets)
                optimizer.zero_grad()
            else:
                with no_grad():
                    datas = datas
                    targets = targets

            instanceOutput, bagOutput, _, _ = modelMIL(datas)

            maxPrediction, index = max(instanceOutput, 0)

            lossBag = criterion(bagOutput.view(1, -1), max(targets.view(1, -1), 1)[1])
            lossMax = criterion(maxPrediction.view(1, -1), max(targets.view(1, -1), 1)[1])

            loss_total = (lossBag + lossMax) / 2
            loss_total = loss_total.mean()

            if training:
                loss_total.backward()
                optimizer.step()

            instanceBagDic[ID][label]["instanceOutput"] = instanceOutput.detach().numpy()
            lossSum += loss_total.cpu().item()
            labelClasses = max(targets, 1)[1].detach().cpu().numpy()
            predClasses = max(bagOutput, 1)[1].detach().cpu().numpy()
            correct += (predClasses == labelClasses).sum()

    lossAvg = lossSum / current
    accAvg = correct / current

    print("%s Epoch %d | Avarage Loss %.4f, Average Accuracy %.2f%%"%(status, epoch, lossAvg, accAvg * 100))

    return lossAvg, accAvg, instanceBagDic

def OrganizePatchImageResults(infoDic, dataInfosDic, labels, outputs):

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
        if key == "label":
            label = dataInfosDic["labelInfos"]["label"]
            infoDic[key].extend(label)
        if key == "outputMatrix":
            infoDic[key].extend(outputs.tolist())

    return infoDic

def PatchImageInstanceProcessing(model, device, dataLoader, announceBatchStep=100):

    model.eval()

    correct, current = 0, 0
    accSum, lossSum = 0, 0
    total = len(dataLoader.dataset)
    infos = GetDefaultInfo()

    for _idx, (dataInfosDic, datas, labels) in enumerate(dataLoader):

        datas, labels = datas.to(device), labels.to(device)

        with no_grad():
            datas = datas
            labels = labels

        outputs = model(datas)

        infos = OrganizePatchImageResults(infos, dataInfosDic, labels, outputs)

        current += dataLoader.batch_size

        if _idx%announceBatchStep == 0:
            print("[%d/%d (%.2f%%)]"%(current, total, (100*current/total)), end='\r')

    del dataInfosDic, datas, labels
    current_device()

    return infos
