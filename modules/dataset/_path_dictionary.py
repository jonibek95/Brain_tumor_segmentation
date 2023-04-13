from numpy.random import permutation
from os.path import isdir, join
from os import mkdir, walk


def AddPathDic(pathDicTotal, pathDic, IDSep):

    for _type in IDSep:

        if _type not in pathDicTotal:
            pathDicTotal[_type] = {}

        for ID in IDSep[_type]:
            pathDicTotal[_type][ID] = pathDic[ID]

    return pathDicTotal

def AnnotatedDataPathPack(pathRootWSI, pathRootDiag, pathRootAnno, fileWSITypes=(".svs", ".mrxs"), fileDiagTypes=(".png", ".tiff"), fileAnnoTypes=(".png", ".tiff")):
    
    pathDicWSI = ReadFilesSubDir(pathRootWSI, fileWSITypes)
    pathDicDiag = ReadFilesSubDir(pathRootDiag, fileDiagTypes)
    pathDicAnno = ReadFilesSubDir(pathRootAnno, fileAnnoTypes)
    
    pathDic = {}
    for ID in pathDicWSI:
        if ID in pathDicDiag and ID in pathDicAnno:
            pathDic[ID] = {}
            pathDic[ID]["pathWSI"] = pathDicWSI[ID]
            pathDic[ID]["pathDiag"] = pathDicDiag[ID]
            pathDic[ID]["pathAnno"] = pathDicAnno[ID]
    
    return pathDic

def CheckNMakeDirectoryTree(pathTree):

    pathTreeSplit = pathTree.split('/')
    pathOutputStep = ""
    for pathEach in pathTreeSplit[:-1]:
        pathOutputStep += pathEach + '/'
        if not isdir(pathOutputStep):
            mkdir(pathOutputStep)

    return None

def LabeledDataPathPack(pathRootWSI, pathRootDiag, classDic, fileWSITypes=(".svs", ".mrxs"), fileDiagTypes=(".png", ".tiff")):
    
    pathDicWSI = ReadFilesSubDir(pathRootWSI, fileWSITypes)
    pathDicDiag = ReadFilesSubDir(pathRootDiag, fileDiagTypes)
    
    pathDic = {}
    for ID in pathDicWSI:
        if ID in pathDicDiag and ID in classDic:
            pathDic[ID] = {}
            pathDic[ID]["pathWSI"] = pathDicWSI[ID]
            pathDic[ID]["pathDiag"] = pathDicDiag[ID]
            pathDic[ID]["label"] = classDic[ID]
    
    return pathDic

def RatioSeparation(ratioDic, unclassifiedData):

    total = 0
    for _type in ratioDic:
        total += ratioDic[_type]

    ratioNormalize = {}
    for  _type in ratioDic:
        valueNormalize = ratioDic[_type] / total
        if valueNormalize > 0:
            ratioNormalize[_type] = valueNormalize

    IDList = permutation(unclassifiedData)
    total = len(IDList)
    start = 0

    dataSepRatio = {}
    for  _type in ratioNormalize:
        end = start + ratioNormalize[_type] * total
        dataSepRatio[_type] = IDList[int(start):int(end)].tolist()
        start = end

    return dataSepRatio

def ReadFilesSubDir(rootPath, fileTypes):

    fileDic = {}
    for (filePath, dir, files) in walk(rootPath, followlinks=True):
        for fileName in files:
            ID = fileName
            ext = fileName.split('.')[-1]
            for fileType in fileTypes:
                if fileType in ID:
                    ID = ID.rpartition(fileType)[0]
                    if ('.'+ext).lower() in fileType:
                        fileDic[ID] = join(filePath, fileName)

    return fileDic
