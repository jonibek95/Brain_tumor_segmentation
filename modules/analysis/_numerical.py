from numpy import exp, sum


def ConvertLabelNOutput2Softmax(outputs, labels, predictions):
    
    probablityOriginalPredictionList = []
    for output, label, prediction in zip(outputs, labels, predictions):
        probablityOriginalPredictionList.append((exp(output[[label, prediction]]) / sum(exp(output))).tolist())

    return probablityOriginalPredictionList