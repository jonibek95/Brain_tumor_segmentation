from torch.utils.data import Dataset, DataLoader
from numpy import zeros, array, mean
from PIL.Image import fromarray


def MakeLabelFromAnnotationArray(handle, location, size, allowRatio=0.3):

    mask = handle[location[1]:location[1]+size[1], location[0]:location[0]+size[0]]
    mat = zeros(2)
    if mean(mask) > allowRatio:
        mat[0] = 1
    else:
        mat[1] = 1

    return mat

def MakeLabelFromListArray(label, labels):

    mat = zeros(len(labels))
    mat[labels.index(label)] = 1

    return mat

class PatchImageDatasetLabel(Dataset):

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

        if "label" in self.dataInfosDic["labelInfos"]:
            label = self.dataInfosDic["labelInfos"]["label"]
            labels = self.dataInfosDic["labelInfos"]["labels"]
            self.Label = MakeLabelFromListArray(label, labels)
        else:
            locationAnno = self.dataInfosDic["labelInfos"]["location"]
            sizeAnno = self.dataInfosDic["labelInfos"]["size"]
            self.Label = MakeLabelFromAnnotationArray(self.ID2Handle[ID]["pathAnno"], locationAnno, sizeAnno)

        return self.dataInfosDic, self.data, self.Label

    def __len__(self):
        return len(self.dataInfosDictionaries)