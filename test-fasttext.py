import fasttext
from sklearn.mixture import GaussianMixture
import re
import numpy as np

trainFile = "train-wiki-new.tsv"
fastTextModel = "/media/4TB/alessio/ai4eu/cc.it.300.bin"
ne_re = re.compile(r"^(.*)\s([^\s]+)$")

task_data_vocabulary = set()

print("Loading dataset")
with open(trainFile, "r") as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        m = ne_re.match(line)
        task_data_vocabulary.add(m.group(1))
print(len(task_data_vocabulary), "words in vocabulary")

print("Loading fasttext model")
model = fasttext.load_model(fastTextModel)

print("Fitting GaussianMixture")
H = np.array([model[w] for w in task_data_vocabulary])
clustering = GaussianMixture(100, covariance_type='diag')
H_clustered = clustering.fit_predict(H)

print(model['slkdfjdfs'][:10])
print(clustering.predict(model['slkdfjdfs'])[:10])
# print(len(H_clustered[0]))


# from collections import OrderedDict
 
# dict = {'ravi':'10','rajnish':'9','sanjeev':'15','yash':'2','suraj':'32'}
# dict1 = OrderedDict(sorted(dict.items()))
# print(dict1)
