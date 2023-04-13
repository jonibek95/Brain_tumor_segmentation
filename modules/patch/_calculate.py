def ConvertMicron2Pixel(micronSizes, mpp):
    return (int(micronSizes[0] / mpp['x']), int(micronSizes[1] / mpp['y']))