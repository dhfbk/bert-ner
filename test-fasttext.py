import argparse
import os

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)
parser = argparse.ArgumentParser(description='Train/test a NER model with FastText.', formatter_class=formatter)
parser.add_argument('--model', help="Model folder (required)", required=True, metavar='FOLDER')
parser.add_argument('--train_file', help="TSV file containing train data", metavar="FILE")
parser.add_argument('--test_file', help="TSV file containing test data", metavar="FILE")
parser.add_argument('--use_gmm', help="Use GaussianMixture (train only)", action='store_true')
parser.add_argument('--use_fasttext', help="Use FastText (train and test)", metavar="FASTTEXT_MODEL")
parser.add_argument('--use_gazette', help="Use gazette (train only)", nargs='+', metavar="GAZ_FILE")
args = parser.parse_args()

trainFile = args.train_file
testFile = args.test_file
modelFolder = args.model

useFastText = args.use_fasttext is not None
useGMM = args.use_gmm
useGazette = args.use_gazette is not None
gazetteFiles = args.use_gazette
fastTextModel = args.use_fasttext

if trainFile is None and testFile is None:
    parser.error("You must specify at least one option between --train_file or --test_file")
if trainFile is None and not os.path.exists(modelFolder):
    parser.error("Model folder must exist when using --test_file alone")
if useGMM and not useFastText:
    parser.error("GMM can be used only with FastText")

gm_file = os.path.join(modelFolder, "gmm.ser")
crf_file = os.path.join(modelFolder, "crf.ser")
gaz_file = os.path.join(modelFolder, "gaz.ser")
ft_file = os.path.join(modelFolder, "ft")

if testFile is not None and os.path.exists(ft_file) and not useFastText:
    parser.error("You are testing a model that uses FastText without including path for it")

# print(args)
# exit()

import fasttext
from sklearn.mixture import GaussianMixture
import re
import numpy as np
import pickle
import tqdm

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import precision_recall_fscore_support

from utils import loadDataset, sent2features, sent2labels

ne_re = re.compile(r"^(.*)\s([^\s]+)$")

ftModel = None
gmClustering = None
gazette = {}
gazetteL = {}

if trainFile is not None:

    if not os.path.exists(modelFolder):
        os.mkdir(modelFolder)

    if useGazette:
        print("Loading gazette")
        if os.path.exists(gaz_file):
            gazette, gazetteL = pickle.load(open(gaz_file, 'rb'))
        else:
            for g in gazetteFiles:
                with open(g, "r") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        parts = re.split(r"\s+", line)
                        label = parts[0]
                        if label not in gazette:
                            gazette[label] = set()
                        if label not in gazetteL:
                            gazetteL[label] = set()
                        for w in parts[1:]:
                            gazette[label].add(w)
                            gazetteL[label].add(w.lower())
            pickle.dump((gazette, gazetteL), open(gaz_file, 'wb'))

    if useFastText:
        print("Loading fasttext model")
        ftModel = fasttext.load_model(fastTextModel)
        with open(ft_file, "w") as fw:
            fw.write("1\n")

    print("Loading dataset")
    train_data = loadDataset(trainFile, ne_re)

    task_data_vocabulary = set()
    for sent in train_data:
        for token in sent:
            task_data_vocabulary.add(token[0])
    print(len(task_data_vocabulary), "words in vocabulary")

    if useGMM:
        if os.path.exists(gm_file):
            print("Loading GaussianMixture")
            gmClustering = pickle.load(open(gm_file, 'rb'))
        else:
            print("Fitting GaussianMixture")
            H = np.array([ftModel[w] for w in task_data_vocabulary])
            gmClustering = GaussianMixture(100, covariance_type='diag', max_iter=200, n_init=3)
            gmClustering.fit_predict(H)
            pickle.dump(gmClustering, open(gm_file, 'wb'))

    print("Extracting features")
    X_train = [sent2features(s, ftModel = ftModel, gmClustering = gmClustering, gazette = gazette, gazetteL = gazetteL) for s in tqdm.tqdm(train_data)]
    y_train = [sent2labels(s) for s in train_data]

    if not os.path.exists(crf_file):
        print("Training model")
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            model_filename=crf_file
        )
        crf.fit(X_train, y_train)
    else:
        print("Loading model")
        crf = sklearn_crfsuite.CRF(model_filename=crf_file)

if testFile is not None:

    # TODO: Save this information in model folder
    if os.path.exists(ft_file):
        print("Loading fasttext model")
        ftModel = fasttext.load_model(fastTextModel)

    if os.path.exists(gm_file) and gmClustering is None:
        print("Loading GaussianMixture")
        gmClustering = pickle.load(open(gm_file, 'rb'))

    if os.path.exists(gaz_file):
        gazette, gazetteL = pickle.load(open(gaz_file, 'rb'))
        print("Loading gazette")

    print("Loading dataset")
    test_data = loadDataset(testFile, ne_re)

    print("Extracting features")
    X_test = [sent2features(s, ftModel = ftModel, gmClustering = gmClustering, gazette = gazette, gazetteL = gazetteL) for s in tqdm.tqdm(test_data)]
    y_test = [sent2labels(s) for s in test_data]

    print("Loading model")
    crf = sklearn_crfsuite.CRF(model_filename=crf_file)

    print("Evaluating model")
    okLabels = list(crf.classes_)
    okLabels.remove('O')
    y_pred = crf.predict(X_test)

    y_test_flat = []
    for sublist in y_test:
        y_test_flat.extend(sublist)
    y_pred_flat = []
    for sublist in y_pred:
        y_pred_flat.extend(sublist)

    print("Macro:", precision_recall_fscore_support(y_test_flat, y_pred_flat, average='macro', labels=okLabels)[:3])
    print("Micro:", precision_recall_fscore_support(y_test_flat, y_pred_flat, average='micro', labels=okLabels)[:3])
    print("Weighted:", precision_recall_fscore_support(y_test_flat, y_pred_flat, average='weighted', labels=okLabels)[:3])
    results = precision_recall_fscore_support(y_test_flat, y_pred_flat, average=None, labels=okLabels)
    support = results[len(results) - 1]
    results = np.delete(results, len(results) - 1, axis=0)
    results = np.transpose(results)
    for i, l in enumerate(okLabels):
        print(l + ":", results[i])
    print("Support:", support)

