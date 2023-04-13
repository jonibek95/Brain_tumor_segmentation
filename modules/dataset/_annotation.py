from torch.utils.data import Dataset, DataLoader
from numpy.random import choice
from numpy import float32, round, array, nan_to_num
from PIL.Image import fromarray
from cv2 import ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, ROTATE_180, rotate, resize, flip


def MakeAnnotationArray(handle, location, size, sizeData, allowRatio=0.3):

    x, y = location
    w, h = size
    mat = handle[y:y+h,x:x+w]
    mat = resize(mat, dsize=(sizeData[0], sizeData[1]))

    return mat

def DeformMatrix(matrix, deformKey):

    infoFlipW = deformKey["flipW"]
    infoFlipH = deformKey["flipH"]
    infoAngleD = deformKey["rotation"]

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

    if infoFlipW:
        matrix = flip(matrix, 1)
    if infoFlipH:
        matrix = flip(matrix, 0)

    return matrix

class PatchImageDatasetAnnotation(Dataset):

    def __init__(self, inputDic, transform=None):
        self.dataInfosDictionaries = inputDic["dataInfosDictionaries"]
        self.ID2Handle = inputDic["ID2Handle"]
        self.labelSize = inputDic["labelSize"]
        self.transform = transform

    def __getitem__(self, index):
        self.dataInfosDic = self.dataInfosDictionaries[index]
        ID = self.dataInfosDic["ID"]

        patchInfos = self.dataInfosDic["patchInfos"]
        location = patchInfos["location"]
        level = patchInfos["level"]
        size = patchInfos["size"]
        self.data = array(self.ID2Handle[ID]["pathWSI"].read_region(location, level, size).convert("RGB"))

        locationAnno = self.dataInfosDic["labelInfos"]["location"]
        sizeAnno = self.dataInfosDic["labelInfos"]["size"]
        self.label = MakeAnnotationArray(self.ID2Handle[ID]["pathAnno"], locationAnno, sizeAnno, self.labelSize) / 255

        infoAngleD = choice([0, 90, 180, 270])
        infoFlipWH = choice([True, False], 2)
        self.deformKey = {"flipW" : infoFlipWH[0], "flipH" : infoFlipWH[1], "rotation" : infoAngleD}

        self.data = DeformMatrix(self.data, self.deformKey)
        self.label = DeformMatrix(self.label, self.deformKey)
        self.data = self.transform(fromarray(self.data))
        
        return self.dataInfosDic, self.data, round(self.label).astype(float32), self.deformKey

    def __len__(self):
        return len(self.dataInfosDictionaries)