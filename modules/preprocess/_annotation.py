import xml.etree.ElementTree
from numpy import round, zeros, array, int32, uint8
import fiona
from cv2 import fillPoly


def ConvertPolygon2Mask(ratioDown, boundDic, annotationCorrdinatesList):
    
    mask = zeros((int(boundDic['h'] / ratioDown), int(boundDic['w'] / ratioDown)), dtype=bool)
    for annotationCorrdinates in annotationCorrdinatesList:

        annotationCorrdinatesMini = round(array(annotationCorrdinates) / ratioDown).astype(int32)

        annotationCorrdinatesMini[annotationCorrdinatesMini < 0] = 0
        maxX, minX = max(annotationCorrdinatesMini[:,0]), min(annotationCorrdinatesMini[:,0])
        maxY, minY = max(annotationCorrdinatesMini[:,1]), min(annotationCorrdinatesMini[:,1])
        annotationCorrdinatesMini -= array([minX, minY])
        mask[minY:maxY,minX:maxX] += fillPoly(zeros((maxY - minY, maxX - minX)), [annotationCorrdinatesMini], (1)).astype(bool)

    mask = mask.astype(uint8)*255
    
    return mask

def OrganizePloygon(annotationPath):

    polygonDic = {}

    tree = xml.etree.ElementTree.parse(annotationPath)
    root = tree.getroot()

    for annotation in root.find("Annotations").findall("Annotation"):
        annoType = annotation.get("class")
        if ',' in annoType:
            annoTypes = annoType
            for annoType in annoTypes.split(','):
                if annoType not in polygonDic:
                    polygonDic[annoType] = []
                corrdinateList = []
                for coordinate in annotation.find("Coordinates").findall("Coordinate"):
                    corrdinateList.append(tuple((float(coordinate.get('x')), float(coordinate.get('y')))))
                polygonDic[annoType].append(corrdinateList)
        else:
            if annoType not in polygonDic:
                polygonDic[annoType] = []
            corrdinateList = []
            for coordinate in annotation.find("Coordinates").findall("Coordinate"):
                corrdinateList.append(tuple((float(coordinate.get('x')), float(coordinate.get('y')))))
            polygonDic[annoType].append(corrdinateList)

    return polygonDic

def MakeAnnotationCorrdinate(annotationPath, typeParse):
    
    if typeParse == "aperio_xml":
        annotationCorrdinatesList = ParserAperioXml(annotationPath)
    elif typeParse == "qupath_geojson":
        annotationCorrdinatesList = ParserQuPathGeojson(annotationPath)
    else:
        raise ValueError("Undefined parser.")

    return annotationCorrdinatesList

def MakeAnnotationMask(ratioDown, boundDic, annotationPath, typeParse):

    mask = zeros((int(boundDic['h'] / ratioDown), int(boundDic['w'] / ratioDown)), dtype=bool)

    annotationCorrdinatesList = MakeAnnotationCorrdinate(annotationPath, typeParse)
    for annotationCorrdinates in annotationCorrdinatesList:
        annotationCorrdinatesMini = (annotationCorrdinates / array([ratioDown, ratioDown])).astype(int32)
        annotationCorrdinatesMini[annotationCorrdinatesMini < 0] = 0
        maxX, minX = max(annotationCorrdinatesMini[:,0]), min(annotationCorrdinatesMini[:,0])
        maxY, minY = max(annotationCorrdinatesMini[:,1]), min(annotationCorrdinatesMini[:,1])
        annotationCorrdinatesMini -= array([minX, minY])
        mask[minY:maxY,minX:maxX] += fillPoly(zeros((maxY - minY, maxX - minX)), [annotationCorrdinatesMini], (1)).astype(bool)

    mask = mask.astype(uint8)*255

    return mask

def ParserAperioXml(annotationPath):
    
    tree = xml.etree.ElementTree.parse(annotationPath)
    root = tree.getroot()
    annotationCorrdinatesList = []
    for _tag_region in root.find("Annotation").find("Regions").findall("Region"):
        annotationCorrdinates = []
        for tag_vertex in _tag_region.find("Vertices").findall("Vertex"):
            annotationCorrdinates.append((float(tag_vertex.get('X')), float(tag_vertex.get('Y'))))
        annotationCorrdinatesList.append(annotationCorrdinates)

    return annotationCorrdinatesList

def ParserQuPathGeojson(annotationPath):
    
    annotationCorrdinatesList = []
    for annotationCorrdinates in fiona.open(annotationPath):
        annotationCorrdinatesList.append(annotationCorrdinates["geometry"]["coordinates"][0])

    return annotationCorrdinatesList