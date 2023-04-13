def GetBoundDic(OS):

    PROPERTY_NAME_BOUNDS_X = "openslide.bounds-x"
    if PROPERTY_NAME_BOUNDS_X in OS.properties:
        PROPERTY_NAME_BOUNDS_Y = "openslide.bounds-y"
        PROPERTY_NAME_BOUNDS_WIDTH = "openslide.bounds-width"
        PROPERTY_NAME_BOUNDS_HEIGHT = "openslide.bounds-height"
        boundsX = int(OS.properties[PROPERTY_NAME_BOUNDS_X])
        boundsY = int(OS.properties[PROPERTY_NAME_BOUNDS_Y])
        boundsW = int(OS.properties[PROPERTY_NAME_BOUNDS_WIDTH])
        boundsH = int(OS.properties[PROPERTY_NAME_BOUNDS_HEIGHT])
    else:
        boundsX = 0
        boundsY = 0
        boundsW = OS.level_dimensions[0][0]
        boundsH = OS.level_dimensions[0][1]

    return {'x' : boundsX, 'y' : boundsY, 'w' : boundsW, 'h' : boundsH}

def GetMicronPerPixelDic(OS):

    mppX = float(OS.properties["openslide.mpp-x"])
    mppY = float(OS.properties["openslide.mpp-y"])

    return {'x' : mppX, 'y' : mppY}