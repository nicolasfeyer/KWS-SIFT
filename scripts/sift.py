import gc
import pickle
import os
from typing import List

import psutil
from cyvlfeat.sift import dsift
from cyvlfeat import kmeans as km
import numpy as np
from tqdm import tqdm
from scripts.utils import draw_visual_words, to_gray_scale, convert_orig_coord_to_dsift_space, draw_key_points


def get_and_labelize_sift_descriptors(corpus_name: str, images, sift_step: int,
                                      scales: List[int], magnitude_thresholds: List[float], K: int, centroids=None,
                                      to_pickle=False, write_descr=False, write_vw=False, filename_prefix=""):
    cells = {}
    key_points = {}

    if not isinstance(images, list):
        images = [images]

    page_no = 0
    # compute the sift descriptors for each
    for image in tqdm(images):

        # convert to gray scale
        image_grey = to_gray_scale(image)

        kps = dict()
        des = dict()

        # compute the sift descriptors for each feature size
        for scale in scales:
            # img = gaussian(image_grey, sigma=math.sqrt((scale / sift_size) ** 2 - .25))
            # img = gaussian(image_grey, sigma=scale ** 2 - 1 / 4)
            # cv2.imshow(str(scale), img)

            # cv2.imwrite(f"res/{filename_prefix}_scale_{scale}.jpg", img * 255)

            kps_, des_ = dsift(image_grey,
                               step=sift_step,
                               norm=True,
                               fast=True,
                               size=scale,
                               # window_size=int(scale / sift_size),
                               float_descriptors=True
                               )
            # kps.append(kps_)
            # des.append(des_)
            if len(kps_) > 0:
                kps[scale] = np.float32(kps_)
                des[scale] = np.float32(des_)

        # cv2.waitKey(0)

        # add the sift descriptors of each feature size
        # des = np.add(des[0], np.add(des[1], des[2])) / len(scales)
        # kps = np.add(kps[0], np.add(kps[1], kps[2])) / len(scales)

        # stack
        # des = np.hstack((des[0], des[1], des[2]))
        # kps = (kps[0] + kps[1] + kps[2]) / len(scales)

        # des = des[0]
        # kps = kps[0]

        cells[page_no] = des
        key_points[page_no] = kps

        if write_descr:
            for i in range(len(scales)):
                mask = key_points[page_no][scales[i]][:, 2] > magnitude_thresholds[i]
                filtered_kp = key_points[page_no][scales[i]][mask]
                draw_key_points(corpus_name, page_no, images[page_no], filtered_kp, scales[i],
                                "kp_" + str(scales[i]) + "_" + filename_prefix)

        page_no += 1

    print("[CORPUS] Labelize Visual Words")
    if centroids is None:
        filtered_cells = list()
        for i in range(len(scales)):
            kps_by_scale = list()
            cells_by_scale = list()
            for k, v in key_points.items():
                kps_by_scale.append(v[scales[i]])
                cells_by_scale.append(cells[k][scales[i]])
            # create a mask to keep significant descriptors regarding the threshold
            mask = np.concatenate(kps_by_scale, axis=0)[:, 2] > magnitude_thresholds[i]

            filtered_cells.append(np.concatenate(cells_by_scale, axis=0)[mask])

        filtered_cells = np.concatenate(filtered_cells, axis=0)

        print("Amount of descriptors used to cluster: {}".format(len(filtered_cells)))

        # use the significant descriptors to fit the kmeans
        centroids = km.kmeans(filtered_cells,
                              num_centers=K,
                              algorithm="ANN",
                              max_num_comparisons=int(K / 50),
                              initialization="PLUSPLUS")

    # for each page, labelize the sift descriptors
    for page_no, descriptors in cells.items():
        for i in range(len(scales)):
            # if there are no descriptors for this scale, skip (it means that the size is too big for the image)
            if scales[i] in descriptors:
                cells[page_no][scales[i]] = km.kmeans_quantize(descriptors[scales[i]], centroids, algorithm="LLOYD")

                # create a mask to spot the sift descriptors that are not significant
                mask = key_points[page_no][scales[i]][:, 2] <= magnitude_thresholds[i]
                # set the visual words to avoid to K to flag them (K is not a valid visual word because it is the number of clusters)
                cells[page_no][scales[i]][mask] = K

                # change the coordinates of the sift descriptors to match the top left corner of the cell and not the center

                # key_points[page_no][scales[i]][:, 0] -= (scales[i] / 2)
                # key_points[page_no][scales[i]][:, 1] -= (scales[i] / 2)

                dims = convert_orig_coord_to_dsift_space(images[page_no].shape[1],
                                                         images[page_no].shape[0],
                                                         scales[i],
                                                         sift_step, 4, ceiled=True)

                cells[page_no][scales[i]] = cells[page_no][scales[i]].reshape(dims[0], dims[1])
                key_points[page_no][scales[i]] = key_points[page_no][scales[i]].reshape((dims[0], dims[1], 3))

                if write_vw:
                    draw_visual_words(corpus_name, page_no, images[page_no], cells[page_no], key_points[page_no],
                                      scales[i],
                                      sift_step,
                                      K,
                                      filename_prefix)

    gc.collect()
    print("[MEMORY USED]", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")

    if to_pickle:
        with open("/".join(['corpus', corpus_name, 'cells_labeled.pickle']), 'wb') as handle:
            pickle.dump(cells, handle)
        with open("/".join(['corpus', corpus_name, 'key_points.pickle']), 'wb') as handle:
            pickle.dump(key_points, handle)
        with open("/".join(['corpus', corpus_name, 'centroids.pickle']), 'wb') as handle:
            pickle.dump(centroids, handle)

    return cells, key_points
