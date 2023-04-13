from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from openslide import OpenSlide
from pytiff import Tiff
from pickle import load
from torch.utils.data import Dataset, DataLoader
from numpy.random import permutation
from numpy import array
from PIL.Image import fromarray


def AdjustKeyListInDict(dicMain, keyList):
    dicSub = dicMain
    for key in keyList:
        dicSub = dicSub[key]
    return dicSub

def GetID2Handle(ID2Path):

    if type(ID2Path) == str:
        ID2Path = load(open(ID2Path, "rb"))

    ID2Handle = {}
    for ID in ID2Path:
        ID2Handle[ID] = {}
        for key, value in ID2Path[ID].items():
            if key in ("pathWSI", "pathAnno"):
                ext = value.split('.')[-1]
                if ext.lower() in ("svs", "mrxs"):
                    ID2Handle[ID][key] = OpenSlide(value)
                if ext.lower() in ("tiff","tif"):
                    ID2Handle[ID][key] = Tiff(value)

    return ID2Handle

def GetKeys(dic, path=[], result=[], init=False):

    if init:
        path, result = [], []
    for key, value in dic.items():
        path.append(key)
        if isinstance(value, dict):
            GetKeys(value, path=path, result=result)
        else:
            result.append(path.copy())
        path.pop()

    return result

def MakeEqualDatasetList(patchInfosTarget, labels):

    patchInfoskeysList = GetKeys(patchInfosTarget, init=True)

    patchInfosDic = {}
    patchInfosNumDic = {}

    for label in labels:
        patchInfosDic[label] = []
        patchInfosNumDic[label] = 0

    for patchInfoskeys in patchInfoskeysList:
        label = patchInfoskeys[-1]
        patchInfoList = AdjustKeyListInDict(patchInfosTarget, patchInfoskeys)
        patchInfosDic[label].extend(patchInfoList)
        patchInfosNumDic[label] += len(patchInfoList)

    patchInfoNumMin = patchInfosNumDic[min(patchInfosNumDic)]

    patchInfosList = []
    for label in labels:
        patchInfosList.extend(permutation(patchInfosDic[label])[:patchInfoNumMin].tolist())

    return patchInfosList

def MakeWholeDatasetList(patchInfosTarget):

    patchInfoskeysList = GetKeys(patchInfosTarget, init=True)

    patchInfosList = []

    for patchInfoskeys in patchInfoskeysList:
        label = patchInfoskeys[-1]
        patchInfoList = AdjustKeyListInDict(patchInfosTarget, patchInfoskeys)
        patchInfosList.extend(patchInfoList)

    return patchInfosList

def MakePatchImageDataLoader(datasetClass, inputDic, batchSize=8, numWorkers=4, dataShuffle=True, dataTransform=None):
    
    if dataTransform is None:
        
        normalize = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                )

        dataTransform = Compose([
            Resize((256, 256)),
            ToTensor(),
            normalize,
            ])
        
    datasetClass = datasetClass(inputDic, transform=dataTransform)

    dataLoader = DataLoader(datasetClass, batch_size=batchSize, shuffle=dataShuffle, drop_last=False, num_workers=numWorkers)
    
    return dataLoader

class PatchImageDatasetUnknown(Dataset):
    
    def __init__(self, inputDic, transform=None):
        self.dataInfosDictionaries = inputDic["dataInfosDictionaries"]
        self.ID2Handle = inputDic["ID2Handle"]
        self.transform = transform
        
    def __getitem__(self, index):
        self.dataInfosDic = self.dataInfosDictionaries[index]
        ID = self.dataInfosDic["ID"]

        patchInfos = self.dataInfosDic["patchInfos"]
        location = patchInfos["location"]
        level = patchInfos["level"]
        size = patchInfos["size"]
        self.data = array(self.ID2Handle[ID]["pathWSI"].read_region(location, level, size).convert("RGB"))
        self.data = self.transform(fromarray(self.data))

        return self.dataInfosDic, self.data
    
    def __len__(self):
        return len(self.dataInfosDictionaries)