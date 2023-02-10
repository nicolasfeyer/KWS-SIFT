import gc
import os
import pickle
from pathlib import Path
import psutil
import numpy as np
from sklearn import preprocessing


def weight_with_tf_idf(corpus_name, patches, W, K, from_pickle=False, to_pickle=False):
    if from_pickle:
        with open("/".join(['corpus', corpus_name, str(W), 'patches_histogram.pickle']), 'rb') as handle:
            patches = pickle.load(handle)

    # M is the total number of local patches
    M = sum(map(lambda x: x.shape[0], patches.values()))

    # the document frequency corresponds to the number of local patches in the collection that contain each the visual word
    # the resulting vector has thus a dimension equal to the number of visual words K
    df = np.sum([np.count_nonzero(patches[i][:, :K], axis=0) for i in range(len(patches))], axis=0)

    idf = np.log(M / (df + 1))

    # apply the tf-idf weighting for each page
    for page_no, ptchs in patches.items():
        patches[page_no] = np.float32(preprocessing.normalize(ptchs * np.hstack([idf] * 3), norm='l2'))

    folder = "/".join(['corpus', corpus_name, str(W)])
    Path(folder).mkdir(parents=True, exist_ok=True)

    with open(folder + "/" + 'idf.pickle', 'wb') as handle:
        pickle.dump(idf, handle)

    gc.collect()
    print("[MEMORY USED]", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")

    if to_pickle:
        with open(folder + "/" + 'patches_tf_idf.pickle', 'wb') as handle:
            pickle.dump(patches, handle)

    return patches
