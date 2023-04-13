from numpy import argmin, array, abs


def ExtractPatchImage(OS, coordinates, level, patchSizes, boundDic):
    coordinates = (boundDic['x'] + coordinates[0], boundDic['y'] + coordinates[1])
    patchSizes = (int(patchSizes[0] / OS.level_downsamples[level]), int(patchSizes[1] / OS.level_downsamples[level]))
    patchImage = array(OS.read_region(coordinates, level, patchSizes).convert("RGB"))
    return patchImage

def GetWSIThumbnail(OS, levelDownsample, boundDic):
    levelDownsamples = array([int(LDS) for LDS in OS.level_downsamples])
    level = argmin(abs(levelDownsample - levelDownsamples))
    thumbImage = ExtractPatchImage(OS, (0, 0), level, (boundDic['w'], boundDic['h']), boundDic)
    return thumbImage