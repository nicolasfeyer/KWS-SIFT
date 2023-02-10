import gc
import os
import pickle
from pathlib import Path
import psutil
import numpy as np
from sklearn.decomposition import TruncatedSVD


def lsa_transform(corpus_name, patches, W, K, T, x_from_pickle=False, patches_from_pickle=False, to_pickle=False):
    folder = "/".join(['corpus', corpus_name, str(W)])
    Path(folder).mkdir(parents=True, exist_ok=True)

    if patches_from_pickle:
        with open("/".join(['corpus', corpus_name, str(W), 'patches_tf_idf.pickle']), 'rb') as handle:
            patches = pickle.load(handle)

    if x_from_pickle:
        with open(folder + "/" + 'lsa_X.pickle', 'rb') as handle:
            X = np.float32(pickle.load(handle))
    else:
        A = np.float32(np.concatenate([patches[i][:, :K] for i in range(len(patches))]).T)
        lsa_obj = TruncatedSVD(n_components=T, n_iter=100, random_state=42)
        U = np.float32(lsa_obj.fit_transform(A))
        S = np.float32(np.diag(lsa_obj.singular_values_))

        X = np.float32(U.dot(np.linalg.inv(S)))

    for page_no, ptchs in patches.items():
        patches[page_no] = np.float32(np.hstack((ptchs[:, :K].dot(X),
                                                 ptchs[:, K:2 * K].dot(X),
                                                 ptchs[:, 2 * K:3 * K].dot(X))))

    gc.collect()
    print("[MEMORY CURRENTLY USED]", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")

    if to_pickle:
        folder = "/".join(['corpus', corpus_name, str(W)])
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(folder + "/" + 'lsa_X.pickle', 'wb') as handle:
            pickle.dump(X, handle)
        with open(folder + "/" + 'patches_lsa.pickle', 'wb') as handle:
            pickle.dump(patches, handle)
