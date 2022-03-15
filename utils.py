import numpy as np

def word2features(sent, index, ftModel = None, gmClustering = None, gazette = {}, gazetteL = {}):
    word = sent[index][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if ftModel is not None:
        ftVec = ftModel[word]
        for i, num in enumerate(ftVec):
            features['ft_' + str(i)] = num
        if gmClustering is not None:
            gmVec = gmClustering.predict([ftVec]);
            features['gm_' + str(gmVec[0])] = True

    for label in gazette:
        if word in gazette[label]:
            features['gaz_' + label] = True
    for label in gazetteL:
        if word.lower() in gazetteL[label]:
            features['gaz_low_' + label] = True

    if index > 0:
        word1 = sent[index - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
        if ftModel is not None:
            ftVec = ftModel[word1]
            for i, num in enumerate(ftVec):
                features['-1:ft_' + str(i)] = num
            if gmClustering is not None:
                gmVec = gmClustering.predict([ftVec]);
                features['-1:gm_' + str(gmVec[0])] = True
        for label in gazette:
            if word1 in gazette[label]:
                features['-1:gaz_' + label] = True
        for label in gazetteL:
            if word.lower() in gazetteL[label]:
                features['-1:gaz_low_' + label] = True
    else:
        features['BOS'] = True

    if index < len(sent) - 1:
        word1 = sent[index + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
        if ftModel is not None:
            ftVec = ftModel[word1]
            for i, num in enumerate(ftVec):
                features['-1:ft_' + str(i)] = num
            if gmClustering is not None:
                gmVec = gmClustering.predict([ftVec]);
                features['-1:gm_' + str(gmVec[0])] = True
        for label in gazette:
            if word1 in gazette[label]:
                features['+1:gaz_' + label] = True
        for label in gazetteL:
            if word.lower() in gazetteL[label]:
                features['+1:gaz_low_' + label] = True
    else:
        features['EOS'] = True

    return features

def loadDataset(fileName, ne_re):
    my_data = []
    tmp_sent = []
    with open(fileName, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                my_data.append(tmp_sent)
                tmp_sent = []
                continue
            m = ne_re.match(line)
            token = m.group(1)
            label = m.group(2)
            tmp_sent.append((token, label))
    return my_data

def sent2features(sent, ftModel = None, gmClustering = None, gazette = {}, gazetteL = {}):
    sentVecs = [word2features(sent, i, ftModel = ftModel, gmClustering = gmClustering, gazette = gazette, gazetteL = gazetteL) for i in range(len(sent))]
    if ftModel is not None:
        vectors = []
        for token in sentVecs:
            v = [token["ft_" + str(i)] for i in range(ftModel.get_dimension())]
            vectors.append(v)
        sentenceVector = np.average(np.array(vectors), axis = 0)
        for token in sentVecs:
            for i in range(len(sentenceVector)):
                token["sent_" + str(i)] = sentenceVector[i]
    return sentVecs

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]
